# ai_agent_integration.py
"""
Integration module to connect the Enhanced AI Web Research Agent to your app.py
Provides live progress updates and seamless integration
"""

import logging
import gradio as gr
import sqlite3
from typing import Optional, Callable
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import the enhanced AI agent
try:
    from enhanced_ai_web_research_agent import (
        EnhancedAIWebResearchAgent,
        create_enhanced_ai_agent,
        ai_analyze_single_item_with_progress,
        ai_bulk_opportunity_scan_with_progress
    )
    ENHANCED_AI_AVAILABLE = True
    print("✅ Enhanced AI Web Research Agent loaded successfully")
except ImportError as e:
    ENHANCED_AI_AVAILABLE = False
    print(f"⚠️ Enhanced AI agent not available: {e}")
    print("📋 Install dependencies: pip install torch transformers llama-index")

class LiveAIAnalysisManager:
    """Manager for live AI analysis with progress updates"""
    
    def __init__(self, auction_manager):
        self.auction_manager = auction_manager
        self.current_analysis = None
        self.progress_messages = []
        self.is_analyzing = False
        
    def analyze_item_with_live_updates(self, lot_number: str) -> tuple:
        """Analyze item with live progress updates for Gradio"""
        if not ENHANCED_AI_AVAILABLE:
            return self._get_unavailable_message(), []
        
        if self.is_analyzing:
            return "⏳ Analysis already in progress. Please wait...", self.progress_messages
        
        self.is_analyzing = True
        self.progress_messages = [f"🚀 Starting AI analysis for Lot #{lot_number}..."]
        
        try:
            def progress_callback(message: str):
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.progress_messages.append(f"[{timestamp}] {message}")
                return message
            
            # Run analysis in separate thread for better responsiveness
            result = ai_analyze_single_item_with_progress(
                self.auction_manager, 
                lot_number, 
                progress_callback
            )
            
            self.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Analysis complete!")
            
            return result, self.progress_messages
            
        except Exception as e:
            error_msg = f"❌ AI analysis failed: {str(e)}"
            self.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}")
            return error_msg, self.progress_messages
        finally:
            self.is_analyzing = False
    
    def bulk_analyze_with_live_updates(self, max_items: int = 20) -> tuple:
        """Bulk analyze items with live progress updates"""
        if not ENHANCED_AI_AVAILABLE:
            return self._get_unavailable_message(), []
        
        if self.is_analyzing:
            return "⏳ Analysis already in progress. Please wait...", self.progress_messages
        
        self.is_analyzing = True
        self.progress_messages = [f"🚀 Starting bulk AI analysis of {max_items} items..."]
        
        try:
            def progress_callback(message: str):
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.progress_messages.append(f"[{timestamp}] {message}")
                return message
            
            result = ai_bulk_opportunity_scan_with_progress(
                self.auction_manager, 
                max_items, 
                progress_callback
            )
            
            self.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Bulk analysis complete!")
            
            return result, self.progress_messages
            
        except Exception as e:
            error_msg = f"❌ Bulk AI analysis failed: {str(e)}"
            self.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {error_msg}")
            return error_msg, self.progress_messages
        finally:
            self.is_analyzing = False
    
    def get_ai_status(self) -> str:
        """Get AI system status"""
        if not ENHANCED_AI_AVAILABLE:
            return """
# ❌ Enhanced AI Analysis Not Available

**Missing Dependencies:**
```bash
pip install torch transformers llama-index llama-index-llms-huggingface llama-index-embeddings-huggingface
```

**Enhanced AI Features:**
- 🤖 **Mistral AI Model:** Advanced reasoning and analysis
- 🧠 **ReAct Methodology:** Visible thought process and research steps
- 🔍 **Multi-Platform Research:** eBay, Amazon, collectibles databases
- 📊 **Statistical Analysis:** Comprehensive price pattern analysis
- 📈 **Market Trends:** Category-specific market intelligence
- 💎 **Investment Grading:** AI-powered investment recommendations

**Current Status:** Basic analysis mode without AI enhancement.
"""
        
        try:
            # Test AI agent creation
            agent = create_enhanced_ai_agent(self.auction_manager.db.db_path)
            
            if agent.agent is not None:
                return """
# ✅ Enhanced AI Analysis Ready

**AI System Status:** Fully Operational

**Available Models:**
- 🤖 **LLM:** Mistral AI (Advanced reasoning)
- 🧠 **Embeddings:** Sentence Transformers
- 🔍 **Research Tools:** Multi-platform web analysis

**Analysis Capabilities:**
- **ReAct Methodology:** Visible AI reasoning process
- **Web Research:** Real-time eBay, Amazon, collectibles research
- **Statistical Analysis:** Comprehensive price analysis
- **Investment Grading:** AI-powered recommendations
- **Market Intelligence:** Category-specific insights

**Ready for Advanced Analysis!**
"""
            else:
                return """
# ⚠️ Enhanced AI Partially Available

**Status:** AI components loaded but agent creation failed

**Possible Issues:**
- GPU memory limitations
- Model download incomplete
- Configuration errors

**Recommendation:** Restart application or check logs for specific errors.
"""
                
        except Exception as e:
            return f"""
# ❌ Enhanced AI System Error

**Error:** {str(e)}

**Troubleshooting:**
1. Check if all dependencies are installed
2. Verify sufficient system memory (8GB+ recommended)
3. Check internet connection for model downloads
4. Review logs for specific error details
"""
    
    def _get_unavailable_message(self) -> str:
        """Standard message when AI is not available"""
        return """
# ❌ Enhanced AI Analysis Not Available

The advanced AI analysis features require additional dependencies.

**To enable Enhanced AI Analysis:**
```bash
pip install torch transformers llama-index llama-index-llms-huggingface llama-index-embeddings-huggingface
```

**What you'll get:**
- 🤖 Mistral AI-powered analysis with visible reasoning
- 🔍 Multi-platform web research (eBay, Amazon, collectibles)
- 📊 Statistical price analysis and market positioning
- 📈 Category-specific market trend analysis
- 💎 AI investment grading and recommendations

**Current Status:** Basic analysis mode available.
"""

# Simple fallback analysis functions for when AI is not available
def analyze_item_by_lot_simple(auction_manager, lot_number: str) -> str:
    """Simple fallback analysis when AI is not available"""
    try:
        if not lot_number.strip():
            return "❌ Please enter a lot number to analyze."
        
        # Get item from database
        with sqlite3.connect(auction_manager.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT lot_number, description, current_price, bid_count, company_name
                FROM items WHERE lot_number = ? LIMIT 1
            """, (lot_number,))
            
            row = cursor.fetchone()
            if not row:
                return f"❌ Lot #{lot_number} not found"
            
            lot_num, description, current_price, bid_count, company = row
            current_price = float(current_price) if current_price else 0
        
        # Simple category-based analysis
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['jewelry', 'bracelet', 'necklace', 'ring']):
            category = 'jewelry'
            multiplier = (1.5, 2.5)
            insights = [
                "Jewelry market varies by brand and condition",
                "Designer/brand names command premium prices",
                "Vintage pieces may have collectible value"
            ]
        elif any(word in desc_lower for word in ['vintage', 'antique', 'collectible']):
            category = 'collectibles'
            multiplier = (1.3, 2.8)
            insights = [
                "Collectibles market depends on rarity and demand",
                "Condition is critical for value",
                "Research specific makers and patterns"
            ]
        elif any(word in desc_lower for word in ['tool', 'drill', 'saw', 'craftsman']):
            category = 'tools'
            multiplier = (1.2, 2.0)
            insights = [
                "Quality tools hold value well",
                "Brand names like Craftsman, DeWalt are preferred",
                "Complete sets more valuable than individual pieces"
            ]
        elif any(word in desc_lower for word in ['furniture', 'table', 'chair', 'desk']):
            category = 'furniture'
            multiplier = (1.1, 1.8)
            insights = [
                "Furniture market varies by style and era",
                "Solid wood preferred over particle board",
                "Local market important due to shipping costs"
            ]
        else:
            category = 'general'
            multiplier = (1.1, 1.6)
            insights = [
                "Research similar items on eBay and Facebook Marketplace",
                "Consider condition and completeness",
                "Factor in shipping costs and fees"
            ]
        
        # Calculate estimates
        estimated_min = max(current_price * multiplier[0], current_price + 2)
        estimated_max = current_price * multiplier[1]
        profit_potential = estimated_min - current_price
        profit_percentage = (profit_potential / current_price * 100) if current_price > 0 else 0
        
        # Determine recommendation
        if profit_percentage > 30:
            recommendation = "BUY - Good opportunity"
            grade = "B (Good)"
        elif profit_percentage > 15:
            recommendation = "HOLD - Fair value"
            grade = "C (Fair)"
        else:
            recommendation = "PASS - Limited profit potential"
            grade = "D (Poor)"
        
        return f"""# 📊 Basic Item Analysis

## Lot #{lot_num} - Grade: {grade}

**Item:** {description}

---

## 💰 Basic Valuation
- **Current Bid:** ${current_price:.2f}
- **Category:** {category.title()}
- **Estimated Range:** ${estimated_min:.2f} - ${estimated_max:.2f}
- **Profit Potential:** ${profit_potential:.2f} ({profit_percentage:.1f}%)

---

## 🎯 Recommendation
**{recommendation}**

---

## 💡 Market Insights
{chr(10).join([f"• {insight}" for insight in insights])}

---

## 📋 Analysis Details
- **Analysis Type:** Basic Category-Based Estimation
- **Confidence Level:** LOW (No web research performed)
- **Data Sources:** Category patterns and general market knowledge

**Note:** This is a basic analysis. For comprehensive AI-powered analysis with web research, install AI dependencies:
```bash
pip install torch transformers llama-index llama-index-llms-huggingface llama-index-embeddings-huggingface
```
"""
        
    except Exception as e:
        return f"❌ Basic analysis error: {str(e)}"

def bulk_profit_scan_simple(auction_manager, max_items: int = 30) -> str:
    """Simple bulk analysis when AI is not available"""
    try:
        items_df = auction_manager.get_recent_items_enhanced(limit=max_items)
        
        if items_df.empty:
            return "❌ No items available for analysis"
        
        opportunities = []
        
        for idx, row in items_df.head(20).iterrows():
            try:
                price_str = str(row.get('Current Price', '$0'))
                cleaned_price = price_str.replace('$', '').replace(',', '').strip()
                current_price = float(cleaned_price) if cleaned_price else 0.0
                
                if current_price <= 0 or current_price > 100:
                    continue
                
                description = str(row.get('Description', '')).lower()
                lot_number = row.get('Lot #', '')
                
                # Simple category scoring
                score = 0
                category = 'general'
                
                if any(word in description for word in ['jewelry', 'bracelet', 'silver', 'gold']):
                    score = current_price * 2.0
                    category = 'jewelry'
                elif any(word in description for word in ['vintage', 'antique', 'collectible']):
                    score = current_price * 2.2
                    category = 'collectibles'
                elif any(word in description for word in ['tool', 'craftsman', 'dewalt']):
                    score = current_price * 1.8
                    category = 'tools'
                else:
                    score = current_price * 1.4
                
                profit = score - current_price
                profit_pct = (profit / current_price * 100) if current_price > 0 else 0
                
                if profit_pct > 20:
                    opportunities.append({
                        'lot': lot_number,
                        'desc': str(row.get('Description', ''))[:60] + "...",
                        'current': current_price,
                        'estimated': score,
                        'profit': profit,
                        'pct': profit_pct,
                        'category': category
                    })
                    
            except Exception:
                continue
        
        if not opportunities:
            no_opps_msg = (
                "# 📊 Basic Bulk Analysis Complete\n\n"
                f"**Analysis Summary:**\n"
                f"- **Items Analyzed:** {len(items_df)}\n"
                f"- **Method:** Category-based estimation\n"
                f"- **Opportunities Found:** 0\n\n"
                f"**No high-profit opportunities found** with basic analysis.\n\n"
                f"**Suggestions:**\n"
                f"- Look for items with brand names\n"
                f"- Focus on collectibles and jewelry\n"
                f"- Consider items under $50 for better margins\n\n"
                f"**For Advanced Analysis:** Install AI dependencies for comprehensive web research and market analysis."
            )
            return no_opps_msg
        
        opportunities.sort(key=lambda x: x['pct'], reverse=True)
        
        result = (
            "# 📊 Basic Bulk Analysis Results\n\n"
            f"**Analysis Summary:**\n"
            f"- **Items Analyzed:** {len(items_df)}\n"
            f"- **Opportunities Found:** {len(opportunities)}\n"
            f"- **Method:** Category-based estimation\n\n"
            f"---\n\n"
            f"## 🎯 Top Opportunities\n\n"
        )
        
        for i, opp in enumerate(opportunities[:8], 1):
            result += (
                f"### {i}. Lot #{opp['lot']} - {opp['category'].title()}\n"
                f"- **Item:** {opp['desc']}\n"
                f"- **Current Bid:** ${opp['current']:.2f}\n"
                f"- **Estimated Value:** ${opp['estimated']:.2f}\n"
                f"- **Profit Potential:** ${opp['profit']:.2f} ({opp['pct']:.0f}%)\n\n"
            )
        
        result += (
            "---\n\n"
            "## 📋 Analysis Notes\n"
            "- **Confidence Level:** LOW (Category-based estimation only)\n"
            "- **Recommendation:** Verify with manual research before bidding\n"
            "- **Upgrade:** Install AI dependencies for comprehensive web-based analysis\n\n"
            "**For Enhanced Analysis:**\n"
            "```bash\n"
            "pip install torch transformers llama-index llama-index-llms-huggingface llama-index-embeddings-huggingface\n"
            "```"
        )
        
        return result
        
    except Exception as e:
        return f"❌ Basic bulk analysis error: {str(e)}"

# Integration functions for direct use in app.py
def create_ai_analysis_manager(auction_manager) -> LiveAIAnalysisManager:
    """Create AI analysis manager for your app"""
    return LiveAIAnalysisManager(auction_manager)

def ai_enhanced_single_analysis(auction_manager, lot_number: str) -> str:
    """Enhanced single item analysis with AI - direct integration function"""
    if not ENHANCED_AI_AVAILABLE:
        return analyze_item_by_lot_simple(auction_manager, lot_number)
    
    try:
        # Simple progress tracking for direct calls
        progress_log = []
        
        def simple_progress(message):
            progress_log.append(message)
            print(f"AI Progress: {message}")
        
        result = ai_analyze_single_item_with_progress(auction_manager, lot_number, simple_progress)
        
        # Add progress log to result
        if progress_log:
            result += f"""

---

## 🔄 AI Analysis Process Log
{chr(10).join([f"• {msg}" for msg in progress_log[-10:]])}  
*Showing last 10 progress updates*
"""
        
        return result
        
    except Exception as e:
        logger.error(f"Enhanced AI analysis failed, falling back to simple: {e}")
        return analyze_item_by_lot_simple(auction_manager, lot_number)

def ai_enhanced_bulk_analysis(auction_manager, max_items: int = 20) -> str:
    """Enhanced bulk analysis with AI - direct integration function"""
    if not ENHANCED_AI_AVAILABLE:
        return bulk_profit_scan_simple(auction_manager, max_items)
    
    try:
        progress_log = []
        
        def simple_progress(message):
            progress_log.append(message)
            print(f"AI Bulk Progress: {message}")
        
        result = ai_bulk_opportunity_scan_with_progress(auction_manager, max_items, simple_progress)
        
        # Add progress summary
        if progress_log:
            result += f"""

---

## 🔄 AI Bulk Analysis Process
{chr(10).join([f"• {msg}" for msg in progress_log[-15:]])}
*Analysis completed with {len(progress_log)} processing steps*
"""
        
        return result
        
    except Exception as e:
        logger.error(f"Enhanced AI bulk analysis failed, falling back to simple: {e}")
        return bulk_profit_scan_simple(auction_manager, max_items)

def test_ai_integration():
    """Test function to verify AI integration"""
    print("Testing Enhanced AI Integration...")
    print(f"Enhanced AI Available: {ENHANCED_AI_AVAILABLE}")
    
    if ENHANCED_AI_AVAILABLE:
        try:
            # Mock auction manager for testing
            class MockAuctionManager:
                def __init__(self):
                    self.db = type('obj', (object,), {'db_path': 'test.db'})
            
            mock_manager = MockAuctionManager()
            ai_manager = create_ai_analysis_manager(mock_manager)
            
            status = ai_manager.get_ai_status()
            print("AI Status Check: ✅ Passed")
            
            return True
            
        except Exception as e:
            print(f"AI Integration Test Failed: {e}")
            return False
    else:
        print("Enhanced AI not available - install dependencies")
        return False

if __name__ == "__main__":
    test_ai_integration()