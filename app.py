#!/usr/bin/env python3
"""
Enhanced Auction Scraper Web Application
Redesigned for better user flow and intuitive AI analysis
"""

import gradio as gr
import pandas as pd
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Tuple
import os

# Import your existing scraper modules
from earl import EarlsAuctionScraper, DatabaseManager, get_recent_auctions, get_auction_items, search_items
from hibid import EnhancedHiBidScraper

try:
    from ai_agent_integration import ai_enhanced_single_analysis, ai_enhanced_bulk_analysis, ENHANCED_AI_AVAILABLE
except ImportError:
    ENHANCED_AI_AVAILABLE = False
    
    def ai_enhanced_single_analysis(auction_manager, lot_number):
        return f"❌ AI analysis not available for Lot #{lot_number}. Install AI dependencies."
    
    def ai_enhanced_bulk_analysis(auction_manager, max_items):
        return "❌ AI bulk analysis not available. Install AI dependencies."
    
# Import catalog scraper
try:
    from hibid_catalog_scraper import scrape_hibid_catalog_url
    CATALOG_SCRAPER_AVAILABLE = True
except ImportError:
    CATALOG_SCRAPER_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auction_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AuctionAppConfig:
    """Configuration class for the auction application"""
    
    def __init__(self):
        self.db_path = '/tmp/auction_data.db'
        self.max_auctions_per_run = 2
        self.max_items_display = 200    
        self.default_zip_code = "46032"
        self.is_hf_space = os.getenv('SPACE_ID') is not None
        
        print(f"Using database at: {self.db_path}")

config = AuctionAppConfig()

class FixedAuctionManager:
    """Fixed auction manager with improved functionality"""
    
    def __init__(self):
        self.db = DatabaseManager(config.db_path)
        self.last_scrape_time = None
        self.scrape_status = "Ready"
        self.total_items = 0
        self.total_auctions = 0
        self.is_scraping = False
        
        # Initialize with some basic stats
        self._update_stats()
    
    def _update_stats(self):
        """Update internal statistics"""
        try:
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT COUNT(*) FROM items 
                    WHERE datetime(scraped_at) > datetime('now', '-24 hours')
                ''')
                self.total_items = cursor.fetchone()[0]
                
                cursor.execute('''
                    SELECT COUNT(*) FROM auctions 
                    WHERE datetime(scraped_at) > datetime('now', '-24 hours')
                ''')
                self.total_auctions = cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
            self.total_items = 0
            self.total_auctions = 0

    def scrape_auctions(self, zip_code: str = None) -> Tuple[str, str]:
        """Scrape auctions for a given zip code"""
        if self.is_scraping:
            return "Scraping already in progress", "Please wait for current scrape to complete"
        
        self.is_scraping = True
        self.scrape_status = "Scraping in progress..."
        
        try:
            zip_code = zip_code or config.default_zip_code
            logger.info(f"Starting auction scrape for zip code: {zip_code}")
            
            all_items = []
            all_auctions = []
            scrape_log = []
            
            # Scrape Earl's Auctions
            try:
                logger.info("Scraping Earl's Auctions...")
                scrape_log.append("🔍 Starting Earl's Auction scrape...")
                
                earls_scraper = EarlsAuctionScraper(
                    zip_code=zip_code, 
                    db_path=config.db_path
                )
                
                earls_items, earls_auctions = earls_scraper.scrape_and_store_all(
                    include_individual_auctions=True,
                    max_auctions=config.max_auctions_per_run
                )
                
                all_items.extend(earls_items)
                all_auctions.extend(earls_auctions)
                
                scrape_log.append(f"✅ Earl's: {len(earls_items)} items, {len(earls_auctions)} auctions")
                
            except Exception as e:
                error_msg = f"❌ Earl's scrape failed: {str(e)}"
                logger.error(error_msg)
                scrape_log.append(error_msg)
            
            # Scrape HiBid Auctions
            try:
                logger.info("Scraping HiBid Auctions...")
                scrape_log.append("🔍 Starting HiBid scrape...")
                
                hibid_scraper = EnhancedHiBidScraper(
                    zip_code=zip_code,
                    db_path=config.db_path
                )
                
                hibid_items, hibid_auctions = hibid_scraper.scrape_and_store_all(
                    include_individual_auctions=True,
                    max_auctions=config.max_auctions_per_run
                )
                
                all_items.extend(hibid_items)
                all_auctions.extend(hibid_auctions)
                
                scrape_log.append(f"✅ HiBid: {len(hibid_items)} items, {len(hibid_auctions)} auctions")
                
            except Exception as e:
                error_msg = f"❌ HiBid scrape failed: {str(e)}"
                logger.error(error_msg)
                scrape_log.append(error_msg)
            
            # Update stats and status
            self._update_stats()
            self.last_scrape_time = datetime.now()
            
            total_items = len(all_items)
            total_auctions = len(all_auctions)
            
            status_msg = f"✅ Scrape completed! Found {total_items} items from {total_auctions} auctions"
            scrape_log.append(status_msg)
            
            self.scrape_status = f"Last updated: {self.last_scrape_time.strftime('%Y-%m-%d %H:%M:%S')}"
            
            logger.info(status_msg)
            return status_msg, "\n".join(scrape_log)
            
        except Exception as e:
            error_msg = f"❌ Scraping failed: {str(e)}"
            logger.error(error_msg)
            self.scrape_status = f"Error: {str(e)}"
            return error_msg, error_msg
            
        finally:
            self.is_scraping = False

    def get_recent_items_enhanced(self, limit: int = 100, search_query: str = "", 
                                company_filter: List[str] = None, 
                                page: int = 1, page_size: int = 50) -> pd.DataFrame:
        """Get recent items with filtering and search"""
        try:
            offset = (page - 1) * page_size
            
            # Build WHERE conditions - expanded time range to show more items
            where_conditions = [
                "datetime(i.scraped_at) > datetime('now', '-7 days')",  # Expanded to 7 days
                "LENGTH(TRIM(i.description)) > 5"
            ]
            params = []
            
            # Add search query filter
            if search_query and search_query.strip():
                where_conditions.append("i.description LIKE ?")
                params.append(f'%{search_query.strip()}%')
            
            # Add company filter
            if company_filter and len(company_filter) > 0:
                company_filter = [c for c in company_filter if c != 'All Companies']
                if company_filter:
                    placeholders = ','.join(['?' for _ in company_filter])
                    where_conditions.append(f"i.company_name IN ({placeholders})")
                    params.extend(company_filter)
            
            where_clause = " AND ".join(where_conditions)
            
            with sqlite3.connect(self.db.db_path) as conn:
                query = f'''
                    SELECT 
                        i.lot_number,
                        i.description,
                        i.current_price,
                        i.bid_count,
                        i.company_name,
                        i.time_remaining,
                        i.source,
                        GROUP_CONCAT(img.image_url, '|') as image_urls
                    FROM items i
                    LEFT JOIN item_images img ON i.id = img.item_id
                    WHERE {where_clause}
                    GROUP BY i.auction_id, i.lot_number
                    ORDER BY i.scraped_at DESC 
                    LIMIT ? OFFSET ?
                '''
                
                params.extend([limit, offset])
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    return pd.DataFrame(columns=['Lot #', 'Description', 'Current Price', 'Bids', 'Company', 'Time Remaining', 'Images'])
            
            # Process data for display - SIMPLIFIED without buttons
            display_df = pd.DataFrame()
            display_df['Lot #'] = df['lot_number']
            display_df['Description'] = df['description']
            
            # Format prices
            display_df['Current Price'] = df['current_price'].apply(
                lambda x: f"${x:.2f}" if pd.notnull(x) and x > 0 else "No bids"
            )
            
            display_df['Bids'] = df['bid_count'].fillna(0).astype(int)
            display_df['Company'] = df['company_name'].fillna('Unknown')
            display_df['Time Remaining'] = df['time_remaining'].fillna('Check site')
            
            # Create image HTML
            def create_image_html(image_urls):
                if not image_urls or pd.isna(image_urls):
                    return "No images"
                
                urls = [url.strip() for url in str(image_urls).split('|') if url.strip()]
                if not urls:
                    return "No images"
                
                first_url = urls[0]
                return f'<img src="{first_url}" style="max-width: 60px; max-height: 60px; object-fit: cover; border-radius: 4px;" title="{len(urls)} images available"/>'
            
            display_df['Images'] = df['image_urls'].apply(create_image_html)
            
            return display_df
                    
        except Exception as e:
            logger.error(f"Error getting recent items: {e}")
            return pd.DataFrame(columns=['Lot #', 'Description', 'Current Price', 'Bids', 'Company', 'Time Remaining', 'Images'])

    def get_available_companies(self) -> List[str]:
        """Get list of available companies for filtering"""
        try:
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT DISTINCT company_name 
                    FROM items 
                    WHERE datetime(scraped_at) > datetime('now', '-7 days')
                    AND company_name IS NOT NULL 
                    AND company_name != ''
                    ORDER BY company_name
                ''')
                
                companies = [row[0] for row in cursor.fetchall()]
                return ['All Companies'] + companies
                
        except Exception as e:
            logger.error(f"Error getting companies: {e}")
            return ['All Companies']

    def get_auction_summary(self) -> Dict:
        """Get summary statistics for display"""
        try:
            with sqlite3.connect(config.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_items,
                        COUNT(DISTINCT company_name) as companies,
                        AVG(current_price) as avg_price,
                        COUNT(CASE WHEN current_price > 0 THEN 1 END) as items_with_bids
                    FROM items 
                    WHERE datetime(scraped_at) > datetime('now', '-7 days')
                ''')
                
                stats = cursor.fetchone()
                total_items, companies, avg_price, items_with_bids = stats
                
                return {
                    'total_items': total_items or 0,
                    'companies': companies or 0,
                    'avg_price': avg_price or 0,
                    'items_with_bids': items_with_bids or 0,
                    'last_update': self.last_scrape_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_scrape_time else 'Never'
                }
                
        except Exception as e:
            logger.error(f"Error getting auction summary: {e}")
            return {
                'total_items': 0,
                'companies': 0,
                'avg_price': 0,
                'items_with_bids': 0,
                'last_update': 'Error'
            }

# Initialize the auction manager
auction_manager = FixedAuctionManager()

# ==================== AUCTION FINDING FUNCTIONS ====================

def find_auctions_by_zip(zip_code: str) -> Tuple[str, str, pd.DataFrame]:
    """Find auctions by ZIP code"""
    if not zip_code.strip():
        return "❌ Please enter a zip code", "", auction_manager.get_recent_items_enhanced()
    
    status, log = auction_manager.scrape_auctions(zip_code.strip())
    updated_df = auction_manager.get_recent_items_enhanced()
    return status, log, updated_df

def scrape_specific_catalog(catalog_url: str) -> Tuple[str, pd.DataFrame]:
    """Scrape a specific HiBid catalog URL"""
    if not CATALOG_SCRAPER_AVAILABLE:
        return "❌ Catalog scraper not available", pd.DataFrame()
    
    if not catalog_url.strip():
        return "❌ Please enter a catalog URL", pd.DataFrame()
    
    if 'hibid.com' not in catalog_url.lower():
        return "❌ Currently only HiBid catalog URLs are supported", pd.DataFrame()
    
    try:
        result = scrape_hibid_catalog_url(catalog_url.strip(), auction_manager.db.db_path)
        
        if result['success']:
            # Refresh the table with new data
            updated_df = auction_manager.get_recent_items_enhanced()
            
            status = f"""
✅ **Catalog Scraping Complete!**

- **Items Found:** {result['items_found']}
- **Items Stored:** {result['items_stored']}

The items are now available in the table below. Use the 🔍 Analyze buttons to get AI analysis for specific items.
"""
            return status, updated_df
        else:
            return f"❌ Catalog scraping failed: {result['message']}", pd.DataFrame()
            
    except Exception as e:
        return f"❌ Scraping error: {str(e)}", pd.DataFrame()

# ==================== SEARCH AND DISPLAY FUNCTIONS ====================

def search_and_filter_items(search_query: str, company_filter: List[str], max_results: int) -> pd.DataFrame:
    """Handle item search and filtering"""
    if isinstance(company_filter, str):
        company_filter = [company_filter] if company_filter != 'All Companies' else []
    
    df = auction_manager.get_recent_items_enhanced(
        limit=max_results, 
        search_query=search_query,
        company_filter=company_filter
    )
    
    return df

def refresh_all_data() -> Tuple[str, pd.DataFrame, gr.update]:
    """Refresh all displayed data"""
    summary = auction_manager.get_auction_summary()
    items_df = auction_manager.get_recent_items_enhanced()
    companies = auction_manager.get_available_companies()
    
    summary_text = f"""
# 📊 Current Auction Data

**Items Available:** {summary['total_items']:,}
**Companies:** {summary['companies']}
**Items with Bids:** {summary['items_with_bids']}
**Average Price:** ${summary['avg_price']:.2f}

**Last Updated:** {summary['last_update']}
**AI Analysis:** {'✅ Available' if ENHANCED_AI_AVAILABLE else '❌ Not Available'}
"""
    
    return summary_text, items_df, gr.update(choices=companies)

# ==================== STARTUP FUNCTION ====================

def startup_sequence():
    """Check existing data and system status"""
    print("🚀 Starting Bucky's Assistant with AI Analysis...")
    
    try:
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM items")
            total_items = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM items 
                WHERE datetime(scraped_at) > datetime('now', '-7 days')
            """)
            recent_items = cursor.fetchone()[0]
    except:
        total_items = 0
        recent_items = 0
    
    if total_items > 0:
        if recent_items > 0:
            print(f"✅ Found {recent_items} recent items, ready to go!")
            status_msg = f"Found {recent_items} recent items"
        else:
            print(f"✅ Found {total_items} items in database (older data)")
            status_msg = f"Found {total_items} items (use 'Find Auctions' for fresh data)"
        
        if ENHANCED_AI_AVAILABLE:
            print("🤖 AI analysis features are ready!")
        return True, status_msg
    else:
        print("📭 No items in database")
        return False, "No data found - use 'Find Auctions' to get started"

# ==================== MAIN INTERFACE ====================

def create_redesigned_interface():
    """Create the redesigned Gradio interface with better flow"""
    
    # Run startup sequence
    startup_success, startup_message = startup_sequence()
    
    with gr.Blocks(
        title="Bucky's Assistant - Auction Analysis",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        .find-section { background-color: #f8f9fa; border: 2px solid #28a745; border-radius: 8px; padding: 15px; margin-bottom: 20px; }
        .ai-section { background-color: #e8f4f8; border: 2px solid #0066cc; border-radius: 8px; padding: 15px; }
        .status-success { background-color: #d4edda; border: 1px solid #c3e6cb; padding: 10px; border-radius: 5px; }
        .status-warning { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; }
        .dataframe button { margin: 2px; }
        .dataframe img { cursor: pointer; border-radius: 4px; }
        """
    ) as interface:
        
        # Header
        header_class = "status-success" if startup_success else "status-warning"
        header_emoji = "✅" if startup_success else "⚠️"
        
        gr.Markdown(f"""
        # 🔨 Bucky's Assistant - Auction Analysis
        
        **AI-powered auction item analysis and profit optimization**
                    
        **With Mistral AI with LlamaIndex ReAct**
        
        <div class="{header_class}">
        {header_emoji} <strong>Status:</strong> {startup_message}
        </div>
        """, elem_classes=["main-header"])
        
        # ==================== SECTION 1: FIND AUCTIONS ====================
        with gr.Group(elem_classes=["find-section"]):
            gr.Markdown("# 🔍 Find Auctions")
            gr.Markdown("*Get auction items using ZIP code search or specific catalog URLs*")
            
            with gr.Tabs():
                # Tab 1: ZIP Code Search
                with gr.TabItem("🗺️ Search by ZIP Code"):
                    gr.Markdown("**Find auctions near you by ZIP code**")
                    with gr.Row():
                        zip_code_input = gr.Textbox(
                            label="ZIP Code",
                            value=config.default_zip_code,
                            placeholder="Enter ZIP code (e.g., 46032)",
                            scale=2
                        )
                        zip_search_btn = gr.Button("🔍 Find Auctions", variant="primary", scale=1)
                
                # Tab 2: Catalog URL
                with gr.TabItem("🔗 Specific Catalog URL"):
                    gr.Markdown("**Scrape a specific HiBid auction catalog**")
                    with gr.Row():
                        catalog_url_input = gr.Textbox(
                            label="HiBid Catalog URL",
                            placeholder="https://hibid.com/state/catalog/123456/auction-name",
                            value="https://hibid.com/indiana/catalog/648763/consignment-auction",
                            scale=2
                        )
                        catalog_search_btn = gr.Button("🔍 Scrape Catalog", variant="primary", scale=1)
            
            # Search Status and Log
            with gr.Row():
                with gr.Column():
                    search_status = gr.Textbox(
                        label="Search Status",
                        value="Ready to search",
                        interactive=False,
                        max_lines=2
                    )
                with gr.Column():
                    search_log = gr.Textbox(
                        label="Search Log",
                        value="Click 'Find Auctions' or 'Scrape Catalog' to start",
                        interactive=False,
                        max_lines=3
                    )

        # ==================== SECTION 2: BROWSE AND FILTER ====================
        with gr.Group():
            gr.Markdown("# 📋 Browse Items")
            
            # Search and Filter Controls
            with gr.Row():
                with gr.Column(scale=2):
                    item_search_input = gr.Textbox(
                        label="Search Items",
                        placeholder="Search descriptions (e.g., 'tools', 'furniture', 'Samsung')"
                    )
                with gr.Column(scale=2):
                    companies_list = auction_manager.get_available_companies()
                    company_filter = gr.CheckboxGroup(
                        label="Filter by Company",
                        choices=companies_list,
                        value=[]
                    )
                with gr.Column(scale=1):
                    max_results = gr.Slider(
                        label="Max Results",
                        minimum=20,
                        maximum=200,
                        value=100,
                        step=20
                    )
            
            with gr.Row():
                filter_btn = gr.Button("🔍 Apply Filters", variant="secondary")
                refresh_btn = gr.Button("🔄 Refresh All Data", variant="secondary")

        # ==================== SECTION 3: ENHANCED AI ANALYSIS ====================
        if ENHANCED_AI_AVAILABLE:
            with gr.Group(elem_classes=["ai-section"]):
                gr.Markdown("# 🤖 Enhanced AI Analysis")
                gr.Markdown("*Powered by Mistral AI with LlamaIndex ReAct methodology*")
                
                # AI Status
                with gr.Accordion("🔧 AI System Status", open=False):
                    def get_ai_status():
                        try:
                            from ai_agent_integration import create_ai_analysis_manager
                            manager = create_ai_analysis_manager(auction_manager)
                            return manager.get_ai_status()
                        except:
                            return "✅ Enhanced AI components loaded and ready"
                    
                    ai_status_display = gr.Markdown(value=get_ai_status())
                    refresh_status_btn = gr.Button("🔄 Refresh Status")
                
                # Single Item Analysis
                gr.Markdown("## 🔍 Single Item AI Analysis")
                with gr.Row():
                    ai_lot_input = gr.Textbox(
                        label="Lot Number",
                        placeholder="Enter lot number for AI analysis",
                        scale=3
                    )
                    ai_analyze_btn = gr.Button("🤖 AI Analyze", variant="primary", scale=1)
                
                ai_analysis_output = gr.Markdown(
                    value="💡 Enter a lot number above and click 'AI Analyze' for comprehensive analysis."
                )
                
                # Bulk Analysis
                gr.Markdown("## 🎯 Bulk AI Analysis")
                with gr.Row():
                    bulk_ai_limit = gr.Slider(
                        label="Items to Analyze",
                        minimum=5,
                        maximum=50,
                        value=20,
                        step=5,
                        scale=2
                    )
                    bulk_ai_btn = gr.Button("🤖 AI Bulk Scan", variant="primary", scale=1)
                
                bulk_ai_output = gr.Markdown(
                    value="🎯 Click 'AI Bulk Scan' to find opportunities using advanced AI analysis."
                )
        else:
            with gr.Group():
                gr.Markdown("# 🤖 AI Analysis (Not Available)")
                gr.Markdown("""
                **AI analysis features are not currently available.**
                
                To enable AI-powered analysis:
                ```bash
                pip install torch transformers llama-index llama-index-llms-huggingface llama-index-embeddings-huggingface
                ```
                
                **Features you'll get:**
                - 🤖 Mistral AI-powered analysis with visible reasoning
                - 🔍 Multi-platform web research (eBay, Amazon, collectibles)
                - 📊 Statistical price analysis and market positioning
                - 📈 Category-specific market trend analysis
                - 💎 AI investment grading and recommendations
                """)

        # ==================== ITEMS TABLE ====================
        gr.Markdown("## 📋 Auction Items")
        gr.Markdown("*Copy lot numbers from the table below for AI analysis*")
        
        # Get initial data for the table
        try:
            if startup_success:
                initial_items = auction_manager.get_recent_items_enhanced(limit=100)
                # Remove the 'Select' column to avoid button issues
                if not initial_items.empty and 'Select' in initial_items.columns:
                    initial_items = initial_items.drop('Select', axis=1)
            else:
                initial_items = pd.DataFrame(columns=['Lot #', 'Description', 'Current Price', 'Bids', 'Company', 'Time Remaining', 'Images'])
        except Exception as e:
            logger.error(f"Error loading initial items: {e}")
            initial_items = pd.DataFrame(columns=['Lot #', 'Description', 'Current Price', 'Bids', 'Company', 'Time Remaining', 'Images'])
        
        # Items table - FIXED: Assign to variable
        items_table = gr.Dataframe(
            value=initial_items,
            label="Auction Items",
            interactive=False,
            wrap=True,
            datatype=["str", "str", "str", "number", "str", "str", "html"]
        )

        # ==================== SUMMARY SECTION ====================
        with gr.Row():
            with gr.Column():
                summary_display = gr.Markdown(
                    value=f"""
# 📊 Current Data Summary

**Items Available:** {auction_manager.get_auction_summary()['total_items']:,}
**Companies:** {auction_manager.get_auction_summary()['companies']}
**Items with Bids:** {auction_manager.get_auction_summary()['items_with_bids']}

**AI Analysis:** {'✅ Available' if ENHANCED_AI_AVAILABLE else '❌ Not Available'}
""",
                    label="Data Summary"
                )
            
            with gr.Column():
                help_info = gr.Markdown(
                    value="""
# 💡 How to Use

1. **Find Auctions:** Use ZIP code or catalog URL
2. **Browse Items:** Filter and search through results  
3. **Analyze Items:** Copy lot number and paste above
4. **Find Opportunities:** Use bulk scanning tools

**Tips:**
- Look for items with detailed descriptions
- Focus on name brands for better profit margins
- Consider shipping costs in your analysis
""",
                    label="Help"
                )

        # ==================== EVENT HANDLERS ====================
        
        # Find Auctions handlers
        zip_search_btn.click(
            fn=find_auctions_by_zip,
            inputs=[zip_code_input],
            outputs=[search_status, search_log, items_table]
        )
        
        catalog_search_btn.click(
            fn=scrape_specific_catalog,
            inputs=[catalog_url_input],
            outputs=[search_status, items_table]
        )
        
        # Filter and search handlers
        filter_btn.click(
            fn=search_and_filter_items,
            inputs=[item_search_input, company_filter, max_results],
            outputs=[items_table]
        )
        
        refresh_btn.click(
            fn=refresh_all_data,
            outputs=[summary_display, items_table, company_filter]
        )

        # AI Analysis event handlers (only if AI is available)
        if ENHANCED_AI_AVAILABLE:
            refresh_status_btn.click(
                fn=get_ai_status,
                outputs=[ai_status_display]
            )
            
            ai_analyze_btn.click(
                fn=lambda lot_num: ai_enhanced_single_analysis(auction_manager, lot_num),
                inputs=[ai_lot_input],
                outputs=[ai_analysis_output]
            )
            
            bulk_ai_btn.click(
                fn=lambda max_items: ai_enhanced_bulk_analysis(auction_manager, max_items),
                inputs=[bulk_ai_limit],
                outputs=[bulk_ai_output]
            )
        
        # Initialize interface with current data
        interface.load(
            fn=lambda: refresh_all_data(),
            outputs=[summary_display, items_table, company_filter]
        )
    
    return interface

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    logger.info("Starting Enhanced Auction Scraper Web Application")
    
    # Create and launch the interface
    app = create_redesigned_interface()
    
    # Launch configuration
    launch_kwargs = {
        "server_port": 7860,
        "show_error": True
    }

    logger.info("Launching redesigned Gradio interface...")
    print("🌐 Interface will be available at: http://localhost:7860")
    app.launch(**launch_kwargs)