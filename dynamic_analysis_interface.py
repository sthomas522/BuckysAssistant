#!/usr/bin/env python3
"""
Dynamic Analysis Interface
Adds clickable descriptions and modal popups for detailed item analysis
"""

import gradio as gr
import pandas as pd
from typing import Dict, List, Optional
import logging
import json
import html

logger = logging.getLogger(__name__)

class DynamicAnalysisInterface:
    """Enhanced interface with clickable items and modal analysis"""
    
    def __init__(self, auction_manager, agent_integration):
        self.auction_manager = auction_manager
        self.agent_integration = agent_integration
        self.current_analysis = None
        self.analysis_cache = {}  # Cache analyses to avoid re-computation
    
    def create_clickable_items_table(self, items_df: pd.DataFrame) -> pd.DataFrame:
        """Convert the items dataframe to have clickable descriptions"""
        if items_df.empty:
            return items_df
        
        # Create a copy to avoid modifying the original
        enhanced_df = items_df.copy()
        
        # Make descriptions clickable with embedded lot number data
        def make_clickable_description(row):
            lot_number = str(row['Lot #']) if 'Lot #' in row else str(row.name)
            description = str(row['Description'])
            
            # Truncate long descriptions for better display
            display_desc = description[:80] + "..." if len(description) > 80 else description
            
            # Create clickable HTML with data attributes
            clickable_html = f'''
            <span 
                class="clickable-description" 
                data-lot="{lot_number}"
                data-description="{html.escape(description)}"
                style="cursor: pointer; color: #0066cc; text-decoration: underline;"
                onclick="analyzeItem('{lot_number}')"
                title="Click for AI Analysis: {html.escape(description)}"
            >
                🔍 {display_desc}
            </span>
            '''
            return clickable_html
        
        # Apply to the Description column
        if 'Description' in enhanced_df.columns:
            enhanced_df['Description'] = enhanced_df.apply(make_clickable_description, axis=1)
        
        # Add analysis status column
        def add_analysis_indicator(row):
            lot_number = str(row['Lot #']) if 'Lot #' in row else str(row.name)
            
            # Quick profit indicator
            try:
                item_data = self._row_to_item_data(row)
                if item_data:
                    from auction_analysis_agent import quick_profit_check
                    profit_result = quick_profit_check(item_data)
                    
                    if "POTENTIAL PROFIT" in profit_result:
                        return "💰 Profit"
                    elif "OVERPRICED" in profit_result:
                        return "⚠️ High"
                    else:
                        return "📊 Check"
                else:
                    return "❓ Unknown"
            except:
                return "📊 Analyze"
        
        enhanced_df['AI Status'] = enhanced_df.apply(add_analysis_indicator, axis=1)
        
        return enhanced_df
    
    def _row_to_item_data(self, row) -> Dict:
        """Convert a dataframe row to item data format"""
        try:
            return {
                'lot_number': str(row.get('Lot #', 'Unknown')),
                'description': str(row.get('Description', '')),
                'current_price': self.agent_integration._parse_price(row.get('Current Price', '0')),
                'bid_count': int(row.get('Bids', 0)) if pd.notnull(row.get('Bids')) else 0,
                'company_name': str(row.get('Company', '')),
                'image_urls': []
            }
        except Exception as e:
            logger.error(f"Error converting row to item data: {e}")
            return None
    
    def create_analysis_modal(self):
        """Create the modal components for detailed analysis"""
        
        with gr.Column(visible=False) as modal_column:
            with gr.Group():
                gr.Markdown("# 🤖 AI Investment Analysis", elem_classes=["modal-header"])
                
                # Modal content areas
                modal_item_info = gr.Markdown("", label="Item Information")
                modal_analysis_content = gr.Markdown("", label="Analysis Results")
                
                # Action buttons in modal
                with gr.Row():
                    modal_close_btn = gr.Button("❌ Close", variant="secondary", scale=1)
                    modal_refresh_btn = gr.Button("🔄 Refresh Analysis", variant="primary", scale=1)
                    modal_save_btn = gr.Button("💾 Save Analysis", variant="secondary", scale=1)
                
                # Quick actions
                with gr.Row():
                    modal_compare_btn = gr.Button("📊 Compare Similar", scale=1)
                    modal_alert_btn = gr.Button("🔔 Set Alert", scale=1)
                    modal_share_btn = gr.Button("📤 Share Analysis", scale=1)
        
        return {
            'modal_column': modal_column,
            'modal_item_info': modal_item_info,
            'modal_analysis_content': modal_analysis_content,
            'modal_close_btn': modal_close_btn,
            'modal_refresh_btn': modal_refresh_btn,
            'modal_save_btn': modal_save_btn,
            'modal_compare_btn': modal_compare_btn,
            'modal_alert_btn': modal_alert_btn,
            'modal_share_btn': modal_share_btn
        }
    
    def analyze_clicked_item(self, lot_number: str) -> tuple:
        """Analyze an item when clicked and return modal data"""
        try:
            # Check cache first
            if lot_number in self.analysis_cache:
                logger.info(f"Using cached analysis for lot {lot_number}")
                analysis_result = self.analysis_cache[lot_number]
            else:
                logger.info(f"Analyzing clicked item: lot {lot_number}")
                analysis_result = self.agent_integration.analyze_single_item(lot_number)
                self.analysis_cache[lot_number] = analysis_result
            
            # Get item details for the header
            item_data = self.agent_integration._get_item_data(lot_number)
            
            if item_data:
                item_info = f"""
## 📦 Item Details
**Lot Number:** {item_data['lot_number']}  
**Current Bid:** ${item_data['current_price']:.2f}  
**Bid Count:** {item_data['bid_count']}  
**Company:** {item_data['company_name']}  

**Description:** {item_data['description']}
"""
            else:
                item_info = f"**Lot Number:** {lot_number}\n*Item details not available*"
            
            # Show modal and return content
            return (
                gr.update(visible=True),  # Show modal
                item_info,                # Item info
                analysis_result,          # Analysis content
                lot_number               # Store current lot for other actions
            )
            
        except Exception as e:
            logger.error(f"Error analyzing clicked item {lot_number}: {e}")
            return (
                gr.update(visible=True),
                f"**Lot Number:** {lot_number}",
                f"❌ Error analyzing item: {str(e)}",
                lot_number
            )
    
    def close_modal(self):
        """Close the analysis modal"""
        return gr.update(visible=False)
    
    def refresh_analysis(self, lot_number: str):
        """Refresh the analysis for the current item"""
        # Remove from cache to force refresh
        if lot_number in self.analysis_cache:
            del self.analysis_cache[lot_number]
        
        # Re-analyze
        return self.analyze_clicked_item(lot_number)
    
    def create_enhanced_interface_with_modal(self):
        """Create the complete interface with modal functionality"""
        
        # Custom CSS for modal and clickable items
        custom_css = """
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 1000;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .modal-content {
            background: white;
            border-radius: 12px;
            padding: 20px;
            max-width: 900px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        
        .modal-header {
            border-bottom: 2px solid #0066cc;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        .clickable-description:hover {
            background-color: #f0f8ff;
            padding: 2px 4px;
            border-radius: 4px;
        }
        
        .profit-indicator {
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
        }
        
        .profit-high { background-color: #d4edda; color: #155724; }
        .profit-medium { background-color: #fff3cd; color: #856404; }
        .profit-low { background-color: #f8d7da; color: #721c24; }
        
        .quick-stats {
            display: flex;
            gap: 15px;
            margin: 10px 0;
            flex-wrap: wrap;
        }
        
        .stat-box {
            background: #f8f9fa;
            padding: 8px 12px;
            border-radius: 6px;
            border-left: 4px solid #0066cc;
            min-width: 120px;
        }
        """
        
        # JavaScript for handling clicks
        custom_js = """
        function analyzeItem(lotNumber) {
            // This will be handled by Gradio event system
            console.log('Analyzing lot:', lotNumber);
            
            // Trigger the analysis via Gradio
            document.dispatchEvent(new CustomEvent('analyze-item', {
                detail: { lotNumber: lotNumber }
            }));
        }
        
        // Add click handlers for dynamic content
        document.addEventListener('DOMContentLoaded', function() {
            document.addEventListener('click', function(e) {
                if (e.target.classList.contains('clickable-description')) {
                    const lotNumber = e.target.getAttribute('data-lot');
                    if (lotNumber) {
                        // Trigger Gradio event
                        analyzeItem(lotNumber);
                    }
                }
            });
        });
        """
        
        return custom_css, custom_js


def create_enhanced_gradio_interface(auction_manager):
    """Create the enhanced Gradio interface with clickable analysis"""
    
    from agent_integration import AuctionAgentIntegration
    agent_integration = AuctionAgentIntegration(auction_manager)
    dynamic_interface = DynamicAnalysisInterface(auction_manager, agent_integration)
    
    # Get custom CSS and JS
    custom_css, custom_js = dynamic_interface.create_enhanced_interface_with_modal()
    
    with gr.Blocks(
        title="Bucky's Assistant - Enhanced AI Analysis",
        theme=gr.themes.Soft(),
        css=custom_css,
        js=custom_js
    ) as interface:
        
        # Hidden state to track current analysis
        current_lot_state = gr.State("")
        
        gr.Markdown("""
        # 🔨 Bucky's Assistant - Enhanced AI Analysis
        
        **Click any item description below for instant AI analysis!**
        
        ✅ Clickable items, 🤖 AI analysis popup, 💰 Profit indicators, 📊 Market insights
        """, elem_classes=["main-header"])
        
        # Main interface layout
        with gr.Row():
            with gr.Column(scale=3):
                # Search and filter section (simplified)
                with gr.Group():
                    gr.Markdown("### 🔍 Find Auctions")
                    with gr.Row():
                        zip_code_input = gr.Textbox(
                            label="Zip Code",
                            value="46032",
                            placeholder="Enter zip code",
                            scale=2
                        )
                        search_input = gr.Textbox(
                            label="Search Items",
                            placeholder="Search descriptions...",
                            scale=2
                        )
                        search_btn = gr.Button("🔍 Search", variant="primary", scale=1)
                
                # Enhanced items table with clickable descriptions
                gr.Markdown("### 📋 Auction Items - Click Any Description for AI Analysis")
                gr.Markdown("*💰 = Profit opportunity | ⚠️ = Currently overpriced | 📊 = Needs analysis*")
                
                items_table = gr.Dataframe(
                    label="Interactive Auction Items",
                    wrap=True,
                    max_height=600,
                    interactive=False,
                    datatype=["str", "html", "str", "number", "str", "str", "html", "str"]
                )
                
            with gr.Column(scale=1):
                # Quick stats and controls
                with gr.Group():
                    gr.Markdown("### 📊 Quick Stats")
                    quick_stats = gr.Markdown("Loading statistics...")
                
                with gr.Group():
                    gr.Markdown("### 🎯 Bulk Actions")
                    bulk_analyze_btn = gr.Button("🔍 Analyze All Visible Items", variant="secondary")
                    profit_scan_btn = gr.Button("💰 Quick Profit Scan", variant="primary")
                    
                bulk_results = gr.Markdown("", label="Bulk Analysis Results")
        
        # Analysis Modal (initially hidden)
        modal_components = dynamic_interface.create_analysis_modal()
        
        # Event handlers for table interactions
        def update_items_table(search_query=""):
            """Update the items table with clickable descriptions"""
            try:
                # Get items from auction manager
                if search_query.strip():
                    items_df = auction_manager.get_recent_items_enhanced(
                        limit=100, 
                        search_query=search_query
                    )
                else:
                    items_df = auction_manager.get_recent_items_enhanced(limit=100)
                
                # Convert to clickable format
                enhanced_df = dynamic_interface.create_clickable_items_table(items_df)
                
                return enhanced_df
                
            except Exception as e:
                logger.error(f"Error updating items table: {e}")
                return pd.DataFrame()
        
        def handle_item_click(evt: gr.SelectData):
            """Handle when user clicks on an item in the table"""
            try:
                # Get the clicked row data
                if evt.index is not None:
                    row_index = evt.index[0]  # Row index
                    col_index = evt.index[1]  # Column index
                    
                    # Only respond to clicks on the Description column (index 1)
                    if col_index == 1:  # Description column
                        # Get current table data
                        current_df = auction_manager.get_recent_items_enhanced(limit=100)
                        
                        if row_index < len(current_df):
                            lot_number = str(current_df.iloc[row_index]['Lot #'])
                            
                            # Trigger analysis
                            return dynamic_interface.analyze_clicked_item(lot_number)
                
                return (gr.update(visible=False), "", "", "")
                
            except Exception as e:
                logger.error(f"Error handling item click: {e}")
                return (gr.update(visible=False), "", f"Error: {str(e)}", "")
        
        def bulk_profit_scan():
            """Quick profit scan of visible items"""
            try:
                items_df = auction_manager.get_recent_items_enhanced(limit=50)
                opportunities = []
                
                for _, row in items_df.iterrows():
                    item_data = dynamic_interface._row_to_item_data(row)
                    if item_data:
                        from auction_analysis_agent import quick_profit_check
                        result = quick_profit_check(item_data)
                        if "POTENTIAL PROFIT" in result:
                            opportunities.append({
                                'lot': item_data['lot_number'],
                                'desc': item_data['description'][:40] + "...",
                                'result': result
                            })
                
                if opportunities:
                    report = f"💰 **Found {len(opportunities)} Profit Opportunities:**\n\n"
                    for opp in opportunities[:5]:
                        report += f"**Lot {opp['lot']}:** {opp['desc']}\n{opp['result']}\n\n"
                    return report
                else:
                    return "📊 No immediate profit opportunities found. Try individual analysis for detailed insights."
                    
            except Exception as e:
                return f"❌ Error in profit scan: {str(e)}"
        
        def update_quick_stats():
            """Update the quick statistics"""
            try:
                items_df = auction_manager.get_recent_items_enhanced(limit=100)
                
                if items_df.empty:
                    return "📊 No items loaded"
                
                total_items = len(items_df)
                total_value = 0
                profit_opportunities = 0
                
                # Quick analysis of a sample
                for _, row in items_df.head(20).iterrows():  # Sample first 20
                    item_data = dynamic_interface._row_to_item_data(row)
                    if item_data and item_data['current_price'] > 0:
                        total_value += item_data['current_price']
                        
                        try:
                            from auction_analysis_agent import quick_profit_check
                            result = quick_profit_check(item_data)
                            if "POTENTIAL PROFIT" in result:
                                profit_opportunities += 1
                        except:
                            pass
                
                avg_price = total_value / max(1, len([r for _, r in items_df.head(20).iterrows() 
                                                   if dynamic_interface._row_to_item_data(r) and 
                                                   dynamic_interface._row_to_item_data(r)['current_price'] > 0]))
                
                return f"""
**📊 Current Auction Stats**
• **Total Items:** {total_items}
• **Average Price:** ${avg_price:.2f}
• **Profit Opportunities:** {profit_opportunities}/20 sampled
• **Click any description for detailed AI analysis**
"""
                
            except Exception as e:
                return f"❌ Error loading stats: {str(e)}"
        
        # Wire up events
        search_btn.click(
            fn=update_items_table,
            inputs=[search_input],
            outputs=[items_table]
        ).then(
            fn=update_quick_stats,
            outputs=[quick_stats]
        )
        
        items_table.select(
            fn=handle_item_click,
            outputs=[
                modal_components['modal_column'],
                modal_components['modal_item_info'],
                modal_components['modal_analysis_content'],
                current_lot_state
            ]
        )
        
        modal_components['modal_close_btn'].click(
            fn=dynamic_interface.close_modal,
            outputs=[modal_components['modal_column']]
        )
        
        modal_components['modal_refresh_btn'].click(
            fn=dynamic_interface.refresh_analysis,
            inputs=[current_lot_state],
            outputs=[
                modal_components['modal_column'],
                modal_components['modal_item_info'],
                modal_components['modal_analysis_content'],
                current_lot_state
            ]
        )
        
        profit_scan_btn.click(
            fn=bulk_profit_scan,
            outputs=[bulk_results]
        )
        
        # Initialize interface
        interface.load(
            fn=lambda: (update_items_table(), update_quick_stats()),
            outputs=[items_table, quick_stats]
        )
    
    return interface


# Usage example for integration with existing app.py
def integrate_dynamic_analysis(auction_manager):
    """Integration function to add to your existing app.py"""
    
    # Replace your existing create_fixed_interface() call with:
    return create_enhanced_gradio_interface(auction_manager)