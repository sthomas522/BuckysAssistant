# hibid_catalog_scraper.py
"""
Enhanced HiBid scraper that can scrape specific auction catalogs
Designed to work with the exact URLs you provided
"""

import requests
from bs4 import BeautifulSoup
import sqlite3
import json
import re
import time
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class HiBidCatalogScraper:
    """Scraper specifically for HiBid auction catalogs"""
    
    def __init__(self, db_path: str = 'auction_data.db'):
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    
    def scrape_catalog_from_url(self, catalog_url: str) -> List[Dict]:
        """
        Scrape a specific HiBid catalog URL
        Example: https://hibid.com/indiana/catalog/648763/consignment-auction
        """
        try:
            logger.info(f"Scraping HiBid catalog: {catalog_url}")
            
            # Extract auction ID from URL
            auction_id_match = re.search(r'/catalog/(\d+)/', catalog_url)
            if not auction_id_match:
                logger.error(f"Could not extract auction ID from URL: {catalog_url}")
                return []
            
            auction_id = auction_id_match.group(1)
            
            # Get the catalog page
            response = self.session.get(catalog_url, timeout=30)
            if response.status_code != 200:
                logger.error(f"Failed to fetch catalog: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract auction metadata
            auction_info = self._extract_auction_info(soup, catalog_url, auction_id)
            
            # Get all catalog items
            items = self._extract_catalog_items(soup, auction_info)
            
            logger.info(f"Found {len(items)} items in catalog {auction_id}")
            return items
            
        except Exception as e:
            logger.error(f"Error scraping catalog {catalog_url}: {e}")
            return []
    
    def _extract_auction_info(self, soup: BeautifulSoup, catalog_url: str, auction_id: str) -> Dict:
        """Extract auction metadata from the catalog page"""
        try:
            auction_info = {
                'auction_id': auction_id,
                'source': catalog_url,
                'company_name': 'Unknown',
                'auction_title': 'Unknown Auction',
                'end_time': None,
                'location': 'Unknown'
            }
            
            # Extract auction title
            title_selectors = [
                'h1.auction-title',
                'h1.page-title', 
                '.auction-header h1',
                'h1',
                '.title'
            ]
            
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    auction_info['auction_title'] = title_elem.get_text().strip()
                    break
            
            # Extract company/auctioneer name
            company_selectors = [
                '.auctioneer-name',
                '.company-name',
                '.auction-company',
                '.auction-header .company'
            ]
            
            for selector in company_selectors:
                company_elem = soup.select_one(selector)
                if company_elem:
                    auction_info['company_name'] = company_elem.get_text().strip()
                    break
            
            # Extract end time if available
            time_selectors = [
                '.auction-end-time',
                '.end-time',
                '.auction-date',
                '[data-end-time]'
            ]
            
            for selector in time_selectors:
                time_elem = soup.select_one(selector)
                if time_elem:
                    # Try to extract time from text or data attribute
                    time_text = time_elem.get_text().strip()
                    if time_text:
                        auction_info['end_time'] = time_text
                        break
                    
                    # Check for data attributes
                    end_time_attr = time_elem.get('data-end-time')
                    if end_time_attr:
                        auction_info['end_time'] = end_time_attr
                        break
            
            # Extract location
            location_selectors = [
                '.auction-location',
                '.location',
                '.address'
            ]
            
            for selector in location_selectors:
                location_elem = soup.select_one(selector)
                if location_elem:
                    auction_info['location'] = location_elem.get_text().strip()
                    break
            
            return auction_info
            
        except Exception as e:
            logger.error(f"Error extracting auction info: {e}")
            return {
                'auction_id': auction_id,
                'source': catalog_url,
                'company_name': 'Unknown',
                'auction_title': 'Unknown Auction',
                'end_time': None,
                'location': 'Unknown'
            }
    
    def _extract_catalog_items(self, soup: BeautifulSoup, auction_info: Dict) -> List[Dict]:
        """Extract all items from the catalog"""
        items = []
        
        try:
            # Multiple selectors for different HiBid layouts
            item_selectors = [
                '.lot-item',
                '.catalog-item',
                '.auction-item',
                '.lot',
                '[data-lot-id]',
                '.item-row'
            ]
            
            item_elements = []
            for selector in item_selectors:
                elements = soup.select(selector)
                if elements:
                    item_elements = elements
                    logger.info(f"Found items using selector: {selector}")
                    break
            
            if not item_elements:
                # Try to find items in a different way - look for lot numbers
                lot_number_elements = soup.find_all(text=re.compile(r'\bLot\s*#?\s*\d+', re.IGNORECASE))
                if lot_number_elements:
                    logger.info(f"Found {len(lot_number_elements)} lot number references")
                    # Try to find parent containers
                    item_elements = []
                    for lot_text in lot_number_elements[:50]:  # Limit to first 50
                        parent = lot_text.parent
                        while parent and parent.name not in ['div', 'article', 'section'] and len(parent.get_text()) < 200:
                            parent = parent.parent
                        if parent and parent not in item_elements:
                            item_elements.append(parent)
            
            # Extract data from each item
            for idx, item_elem in enumerate(item_elements):
                try:
                    item_data = self._extract_single_item(item_elem, auction_info, idx)
                    if item_data and item_data.get('lot_number'):
                        items.append(item_data)
                        
                except Exception as e:
                    logger.error(f"Error extracting item {idx}: {e}")
                    continue
            
            return items
            
        except Exception as e:
            logger.error(f"Error extracting catalog items: {e}")
            return []
    
    def _extract_single_item(self, item_elem, auction_info: Dict, idx: int) -> Optional[Dict]:
        """Extract data from a single catalog item"""
        try:
            item_data = {
                'auction_id': auction_info['auction_id'],
                'company_name': auction_info['company_name'],
                'auction_title': auction_info['auction_title'],
                'source': auction_info['source'],
                'scraped_at': datetime.now().isoformat()
            }
            
            # Extract lot number
            lot_number = self._extract_lot_number(item_elem)
            if not lot_number:
                lot_number = str(idx + 1)  # Fallback to index
            item_data['lot_number'] = lot_number
            
            # Extract description
            description = self._extract_description(item_elem)
            if not description or len(description.strip()) < 5:
                return None  # Skip items without valid descriptions
            item_data['description'] = description
            
            # Extract current price/bid
            current_price = self._extract_current_price(item_elem)
            item_data['current_price'] = current_price
            
            # Extract bid count
            bid_count = self._extract_bid_count(item_elem)
            item_data['bid_count'] = bid_count
            
            # Extract images
            images = self._extract_images(item_elem)
            item_data['images'] = images
            
            # Extract time remaining if available
            time_remaining = self._extract_time_remaining(item_elem)
            item_data['time_remaining'] = time_remaining
            
            # Create item-specific URL if possible
            item_url = self._create_item_url(auction_info['source'], auction_info['auction_id'], lot_number)
            item_data['item_url'] = item_url
            
            return item_data
            
        except Exception as e:
            logger.error(f"Error extracting single item: {e}")
            return None
    
    def _extract_lot_number(self, item_elem) -> str:
        """Extract lot number from item element"""
        try:
            # Multiple strategies to find lot number
            strategies = [
                # Look for specific lot number patterns
                lambda elem: self._find_text_by_pattern(elem, r'(?:Lot\s*#?\s*|Item\s*#?\s*)(\d+)', 1),
                # Look for data attributes
                lambda elem: elem.get('data-lot-id') or elem.get('data-lot-number'),
                # Look for class patterns
                lambda elem: self._extract_from_class(elem, 'lot'),
                # Look in specific elements
                lambda elem: self._find_in_descendants(elem, ['.lot-number', '.lot-id', '.item-number'])
            ]
            
            for strategy in strategies:
                result = strategy(item_elem)
                if result:
                    return str(result).strip()
            
            return ""
            
        except Exception:
            return ""
    
    def _extract_description(self, item_elem) -> str:
        """Extract item description"""
        try:
            # Look for description in various places
            desc_selectors = [
                '.item-description',
                '.lot-description', 
                '.description',
                '.item-title',
                '.lot-title',
                '.title',
                'h3',
                'h4',
                '.item-name'
            ]
            
            for selector in desc_selectors:
                desc_elem = item_elem.select_one(selector)
                if desc_elem:
                    desc_text = desc_elem.get_text().strip()
                    if len(desc_text) > 5:  # Must have meaningful content
                        return desc_text
            
            # Fallback: get all text and try to extract meaningful description
            all_text = item_elem.get_text().strip()
            lines = [line.strip() for line in all_text.split('\n') if line.strip()]
            
            # Find the longest meaningful line (likely the description)
            meaningful_lines = [line for line in lines if len(line) > 10 and not re.match(r'^[\d\.\$,\s]+$', line)]
            if meaningful_lines:
                return meaningful_lines[0]
            
            return ""
            
        except Exception:
            return ""
    
    def _extract_current_price(self, item_elem) -> float:
        """Extract current bid/price"""
        try:
            price_selectors = [
                '.current-bid',
                '.current-price',
                '.bid-amount',
                '.price',
                '.high-bid'
            ]
            
            for selector in price_selectors:
                price_elem = item_elem.select_one(selector)
                if price_elem:
                    price_text = price_elem.get_text().strip()
                    price_match = re.search(r'\$?([\d,]+\.?\d*)', price_text)
                    if price_match:
                        return float(price_match.group(1).replace(',', ''))
            
            # Look for price patterns in text
            all_text = item_elem.get_text()
            price_patterns = [
                r'Current\s*Bid:?\s*\$?([\d,]+\.?\d*)',
                r'High\s*Bid:?\s*\$?([\d,]+\.?\d*)',
                r'Price:?\s*\$?([\d,]+\.?\d*)',
                r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)'
            ]
            
            for pattern in price_patterns:
                match = re.search(pattern, all_text, re.IGNORECASE)
                if match:
                    return float(match.group(1).replace(',', ''))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _extract_bid_count(self, item_elem) -> int:
        """Extract number of bids"""
        try:
            bid_selectors = [
                '.bid-count',
                '.num-bids',
                '.bids'
            ]
            
            for selector in bid_selectors:
                bid_elem = item_elem.select_one(selector)
                if bid_elem:
                    bid_text = bid_elem.get_text().strip()
                    bid_match = re.search(r'(\d+)', bid_text)
                    if bid_match:
                        return int(bid_match.group(1))
            
            # Look for bid patterns in text
            all_text = item_elem.get_text()
            bid_patterns = [
                r'(\d+)\s*bids?',
                r'Bids?:?\s*(\d+)',
                r'(\d+)\s*bidders?'
            ]
            
            for pattern in bid_patterns:
                match = re.search(pattern, all_text, re.IGNORECASE)
                if match:
                    return int(match.group(1))
            
            return 0
            
        except Exception:
            return 0
    
    def _extract_images(self, item_elem) -> List[str]:
        """Extract image URLs"""
        try:
            images = []
            
            # Find all img elements
            img_elements = item_elem.find_all('img')
            
            for img in img_elements:
                src = img.get('src') or img.get('data-src') or img.get('data-lazy')
                if src:
                    # Convert relative URLs to absolute
                    if src.startswith('//'):
                        src = 'https:' + src
                    elif src.startswith('/'):
                        src = 'https://hibid.com' + src
                    
                    # Skip placeholder images
                    if any(skip in src.lower() for skip in ['placeholder', 'loading', 'blank', 'default']):
                        continue
                    
                    images.append(src)
            
            return images[:5]  # Limit to 5 images per item
            
        except Exception:
            return []
    
    def _extract_time_remaining(self, item_elem) -> str:
        """Extract time remaining if available"""
        try:
            time_selectors = [
                '.time-remaining',
                '.ends-in',
                '.countdown'
            ]
            
            for selector in time_selectors:
                time_elem = item_elem.select_one(selector)
                if time_elem:
                    return time_elem.get_text().strip()
            
            return ""
            
        except Exception:
            return ""
    
    def _create_item_url(self, catalog_url: str, auction_id: str, lot_number: str) -> str:
        """Create direct URL to item"""
        try:
            # HiBid URL pattern for individual items
            return f"https://hibid.com/lot/{auction_id}-{lot_number}/"
        except Exception:
            return catalog_url
    
    def _find_text_by_pattern(self, elem, pattern: str, group: int = 0) -> str:
        """Find text matching a regex pattern"""
        try:
            text = elem.get_text()
            match = re.search(pattern, text, re.IGNORECASE)
            return match.group(group) if match else ""
        except Exception:
            return ""
    
    def _extract_from_class(self, elem, keyword: str) -> str:
        """Extract value from class names containing keyword"""
        try:
            classes = elem.get('class', [])
            for cls in classes:
                if keyword in cls.lower():
                    # Try to extract number from class
                    match = re.search(r'(\d+)', cls)
                    if match:
                        return match.group(1)
            return ""
        except Exception:
            return ""
    
    def _find_in_descendants(self, elem, selectors: List[str]) -> str:
        """Find text in descendant elements"""
        try:
            for selector in selectors:
                found = elem.select_one(selector)
                if found:
                    text = found.get_text().strip()
                    if text:
                        return text
            return ""
        except Exception:
            return ""
    
    def store_items_in_database(self, items: List[Dict]) -> int:
        """Store scraped items in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create tables if they don't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        lot_number TEXT,
                        description TEXT,
                        current_price REAL,
                        bid_count INTEGER,
                        company_name TEXT,
                        auction_title TEXT,
                        auction_id TEXT,
                        source TEXT,
                        time_remaining TEXT,
                        end_time TEXT,
                        item_url TEXT,
                        scraped_at TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS item_images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        item_id INTEGER,
                        image_url TEXT,
                        FOREIGN KEY (item_id) REFERENCES items (id)
                    )
                ''')
                
                stored_count = 0
                
                for item in items:
                    try:
                        # Insert item
                        cursor.execute('''
                            INSERT INTO items (
                                lot_number, description, current_price, bid_count,
                                company_name, auction_title, auction_id, source,
                                time_remaining, item_url, scraped_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            item.get('lot_number'),
                            item.get('description'),
                            item.get('current_price', 0),
                            item.get('bid_count', 0),
                            item.get('company_name'),
                            item.get('auction_title'),
                            item.get('auction_id'),
                            item.get('source'),
                            item.get('time_remaining'),
                            item.get('item_url'),
                            item.get('scraped_at')
                        ))
                        
                        item_id = cursor.lastrowid
                        
                        # Insert images
                        for image_url in item.get('images', []):
                            cursor.execute('''
                                INSERT INTO item_images (item_id, image_url)
                                VALUES (?, ?)
                            ''', (item_id, image_url))
                        
                        stored_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error storing item {item.get('lot_number', 'unknown')}: {e}")
                        continue
                
                conn.commit()
                logger.info(f"Stored {stored_count} items in database")
                return stored_count
                
        except Exception as e:
            logger.error(f"Database error: {e}")
            return 0

# Integration function for your app
def scrape_hibid_catalog_url(catalog_url: str, db_path: str = 'auction_data.db') -> Dict:
    """
    Scrape a HiBid catalog URL and store in database
    Returns summary of scraping results
    """
    try:
        scraper = HiBidCatalogScraper(db_path)
        items = scraper.scrape_catalog_from_url(catalog_url)
        
        if not items:
            return {
                'success': False,
                'message': 'No items found in catalog',
                'items_found': 0,
                'items_stored': 0
            }
        
        stored_count = scraper.store_items_in_database(items)
        
        return {
            'success': True,
            'message': f'Successfully scraped and stored {stored_count} items',
            'items_found': len(items),
            'items_stored': stored_count,
            'catalog_url': catalog_url
        }
        
    except Exception as e:
        logger.error(f"Error scraping catalog: {e}")
        return {
            'success': False,
            'message': f'Error: {str(e)}',
            'items_found': 0,
            'items_stored': 0
        }

if __name__ == "__main__":
    # Test with the provided URL
    test_url = "https://hibid.com/indiana/catalog/648763/consignment-auction"
    result = scrape_hibid_catalog_url(test_url)
    print(f"Scraping result: {result}")