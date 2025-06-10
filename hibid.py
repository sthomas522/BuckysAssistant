import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from urllib.parse import urljoin, urlparse
import json
from datetime import datetime, timedelta
import sqlite3
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AuctionItem:
    lot_number: str
    description: str
    current_price: Optional[float]
    price_text: str
    bid_count: int
    source: str
    auction_id: Optional[str] = None
    end_time: Optional[str] = None
    time_remaining: Optional[str] = None
    image_urls: List[str] = None
    auction_title: Optional[str] = None
    company_name: Optional[str] = None
    scraped_at: Optional[str] = None

    def __post_init__(self):
        if self.image_urls is None:
            self.image_urls = []
        if self.scraped_at is None:
            self.scraped_at = datetime.now().isoformat()

@dataclass
class AuctionInfo:
    company_name: str
    company_url: str
    auction_title: str
    dates: str
    location: str
    bidding_notice: str
    zip_code: Optional[str]
    end_time: Optional[str] = None
    time_remaining: Optional[str] = None
    auction_id: Optional[str] = None
    scraped_at: Optional[str] = None

    def __post_init__(self):
        if self.scraped_at is None:
            self.scraped_at = datetime.now().isoformat()

class DatabaseManager:
    def __init__(self, db_path='hibid_auctions.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Auctions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS auctions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    auction_id TEXT UNIQUE,
                    company_name TEXT,
                    company_url TEXT,
                    auction_title TEXT,
                    dates TEXT,
                    location TEXT,
                    bidding_notice TEXT,
                    zip_code TEXT,
                    end_time TEXT,
                    time_remaining TEXT,
                    scraped_at TEXT,
                    UNIQUE(auction_id)
                )
            ''')

            # Items table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lot_number TEXT,
                    description TEXT,
                    current_price REAL,
                    price_text TEXT,
                    bid_count INTEGER,
                    source TEXT,
                    auction_id TEXT,
                    end_time TEXT,
                    time_remaining TEXT,
                    auction_title TEXT,
                    company_name TEXT,
                    scraped_at TEXT,
                    FOREIGN KEY (auction_id) REFERENCES auctions (auction_id)
                )
            ''')

            # Images table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS item_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id INTEGER,
                    image_url TEXT,
                    image_order INTEGER,
                    FOREIGN KEY (item_id) REFERENCES items (id)
                )
            ''')

            conn.commit()

    def save_auction(self, auction: AuctionInfo) -> int:
        """Save auction info to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO auctions
                (auction_id, company_name, company_url, auction_title, dates,
                 location, bidding_notice, zip_code, end_time, time_remaining, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                auction.auction_id, auction.company_name, auction.company_url,
                auction.auction_title, auction.dates, auction.location,
                auction.bidding_notice, auction.zip_code, auction.end_time,
                auction.time_remaining, auction.scraped_at
            ))
            return cursor.lastrowid

    def save_item(self, item: AuctionItem) -> int:
        """Save auction item to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO items
                (lot_number, description, current_price, price_text, bid_count,
                 source, auction_id, end_time, time_remaining, auction_title,
                 company_name, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item.lot_number, item.description, item.current_price,
                item.price_text, item.bid_count, item.source, item.auction_id,
                item.end_time, item.time_remaining, item.auction_title,
                item.company_name, item.scraped_at
            ))

            item_id = cursor.lastrowid

            # Save images
            for i, image_url in enumerate(item.image_urls):
                cursor.execute('''
                    INSERT INTO item_images (item_id, image_url, image_order)
                    VALUES (?, ?, ?)
                ''', (item_id, image_url, i))

            return item_id

    def get_active_auctions(self, zip_code: Optional[str] = None) -> List[Dict]:
        """Get active auctions from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = '''
                SELECT * FROM auctions
                WHERE datetime(scraped_at) > datetime('now', '-24 hours')
            '''
            params = []

            if zip_code:
                query += ' AND zip_code = ?'
                params.append(zip_code)

            query += ' ORDER BY scraped_at DESC'

            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_items_by_auction(self, auction_id: str) -> List[Dict]:
        """Get all items for a specific auction"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT i.*, GROUP_CONCAT(img.image_url) as image_urls
                FROM items i
                LEFT JOIN item_images img ON i.id = img.item_id
                WHERE i.auction_id = ?
                GROUP BY i.id
                ORDER BY CAST(i.lot_number AS INTEGER)
            ''', (auction_id,))

            columns = [desc[0] for desc in cursor.description]
            items = []
            for row in cursor.fetchall():
                item = dict(zip(columns, row))
                if item['image_urls']:
                    item['image_urls'] = item['image_urls'].split(',')
                else:
                    item['image_urls'] = []
                items.append(item)
            return items

class EnhancedHiBidScraper:
    def __init__(self, zip_code=None, db_path='hibid_auctions.db'):
        self.base_url = "https://hibid.com"
        self.indiana_url = "https://hibid.com/indiana"
        self.zip_code = zip_code
        if zip_code:
            self.zip_url = f"https://hibid.com/indiana/auctions?zip={zip_code}"
        else:
            self.zip_url = None

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

        self.db = DatabaseManager(db_path)

    def get_page_content(self, url, retries=3):
        """Fetch page content with retry logic"""
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return None

    def parse_price(self, price_text):
        """Extract numeric price from price text"""
        if not price_text:
            return None

        # Remove common currency symbols and spaces
        cleaned = re.sub(r'[^\d.,]', '', price_text)
        price_match = re.search(r'\d+(?:,\d{3})*(?:\.\d{2})?', cleaned)
        if price_match:
            try:
                return float(price_match.group().replace(',', ''))
            except ValueError:
                return None
        return None

    def parse_time_remaining(self, time_text):
        """Parse time remaining from various formats"""
        if not time_text:
            return None, None

        # Clean up the text
        time_text = re.sub(r'\s+', ' ', time_text.strip())

        # Common patterns for time remaining
        patterns = [
            r'(\d+)d\s*(\d+)h\s*(\d+)m',  # 5d 12h 30m
            r'(\d+)\s*days?\s*(\d+)\s*hours?\s*(\d+)\s*min',
            r'(\d+)h\s*(\d+)m',  # 12h 30m
            r'(\d+)\s*hours?\s*(\d+)\s*min',
            r'(\d+)m',  # 30m
            r'(\d+)\s*min',
            r'Ends?:?\s*(.+)',  # Ends: Dec 15, 2024 3:00 PM
            r'Closing:?\s*(.+)',  # Closing: Dec 15, 2024
        ]

        for pattern in patterns:
            match = re.search(pattern, time_text, re.IGNORECASE)
            if match:
                groups = match.groups()

                if len(groups) == 3:  # days, hours, minutes
                    try:
                        days, hours, minutes = map(int, groups)
                        end_time = datetime.now() + timedelta(days=days, hours=hours, minutes=minutes)
                        return end_time.isoformat(), f"{days}d {hours}h {minutes}m"
                    except ValueError:
                        continue

                elif len(groups) == 2 and any(x in time_text.lower() for x in ['h', 'hour']):  # hours, minutes
                    try:
                        hours, minutes = map(int, groups)
                        end_time = datetime.now() + timedelta(hours=hours, minutes=minutes)
                        return end_time.isoformat(), f"{hours}h {minutes}m"
                    except ValueError:
                        continue

                elif len(groups) == 1:
                    if any(x in time_text.lower() for x in ['m', 'min']):  # minutes only
                        try:
                            minutes = int(groups[0])
                            end_time = datetime.now() + timedelta(minutes=minutes)
                            return end_time.isoformat(), f"{minutes}m"
                        except ValueError:
                            continue
                    else:  # Absolute time
                        try:
                            # Try various date formats
                            date_formats = [
                                "%b %d, %Y %I:%M %p",
                                "%B %d, %Y %I:%M %p",
                                "%m/%d/%Y %I:%M %p",
                                "%m-%d-%Y %I:%M %p",
                                "%Y-%m-%d %H:%M:%S",
                                "%m/%d/%Y %H:%M",
                            ]

                            date_str = groups[0].strip()
                            for fmt in date_formats:
                                try:
                                    end_time = datetime.strptime(date_str, fmt)
                                    remaining = end_time - datetime.now()
                                    if remaining.total_seconds() > 0:
                                        days = remaining.days
                                        hours, remainder = divmod(remaining.seconds, 3600)
                                        minutes, _ = divmod(remainder, 60)
                                        return end_time.isoformat(), f"{days}d {hours}h {minutes}m"
                                    break
                                except ValueError:
                                    continue
                        except:
                            pass

        return None, time_text

    def is_valid_image_url(self, url):
        """Check if URL is a valid image URL"""
        if not url:
            return False

        # Check for common image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
        url_lower = url.lower()

        # Check extension
        if any(ext in url_lower for ext in image_extensions):
            return True

        # Check for image-related keywords in URL
        image_keywords = ['image', 'img', 'photo', 'picture', 'thumb', 'gallery']
        if any(keyword in url_lower for keyword in image_keywords):
            return True

        return False

    def extract_images(self, soup, base_url):
        """Enhanced image extraction with multiple strategies"""
        images = []

        # Strategy 1: Look for all img tags and filter by various criteria
        all_imgs = soup.find_all('img')

        for img in all_imgs:
            # Get src or data-src
            src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')

            if not src:
                continue

            # Convert to absolute URL
            full_url = urljoin(base_url, src)

            # Skip common non-item images
            skip_patterns = [
                'logo', 'banner', 'header', 'footer', 'icon',
                'avatar', 'profile', 'social', 'ad', 'advertisement',
                'placeholder', 'loading', 'spinner', '1x1', 'tracking'
            ]

            if any(pattern in full_url.lower() for pattern in skip_patterns):
                continue

            # Check if it's a valid image URL
            if self.is_valid_image_url(full_url):
                # Check image dimensions (if available) to skip tiny images
                width = img.get('width')
                height = img.get('height')

                if width and height:
                    try:
                        w, h = int(width), int(height)
                        if w < 50 or h < 50:  # Skip very small images
                            continue
                    except ValueError:
                        pass

                # Check CSS classes for lot/item related images
                img_classes = img.get('class', [])
                if isinstance(img_classes, str):
                    img_classes = img_classes.split()

                # Prioritize images with lot/item related classes
                priority_classes = ['lot', 'item', 'product', 'auction', 'photo', 'image', 'gallery']
                has_priority = any(cls.lower() in ' '.join(img_classes).lower() for cls in priority_classes)

                if full_url not in images:
                    if has_priority:
                        images.insert(0, full_url)  # Add priority images first
                    else:
                        images.append(full_url)

        # Strategy 2: Look for background images in CSS
        for element in soup.find_all(attrs={"style": True}):
            style = element.get('style', '')
            bg_match = re.search(r'background-image:\s*url\(["\']?([^"\']+)["\']?\)', style)
            if bg_match:
                bg_url = urljoin(base_url, bg_match.group(1))
                if self.is_valid_image_url(bg_url) and bg_url not in images:
                    images.append(bg_url)

        # Strategy 3: Look for data attributes that might contain image URLs
        for element in soup.find_all():
            if hasattr(element, 'attrs') and element.attrs:
                for attr, value in element.attrs.items():
                    if (isinstance(attr, str) and attr.startswith('data-') and
                        'img' in attr.lower() and isinstance(value, str)):
                        if 'http' in value or value.startswith('/'):
                            img_url = urljoin(base_url, value)
                            if self.is_valid_image_url(img_url) and img_url not in images:
                                images.append(img_url)

        # Strategy 4: Look for JSON-LD or other structured data
        json_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    # Look for image fields
                    for key, value in data.items():
                        if 'image' in key.lower() and isinstance(value, str):
                            img_url = urljoin(base_url, value)
                            if self.is_valid_image_url(img_url) and img_url not in images:
                                images.append(img_url)
            except (json.JSONDecodeError, TypeError):
                continue

        # Clean and deduplicate
        cleaned_images = []
        for img_url in images[:10]:  # Limit to 10 images
            # Clean up URL
            parsed = urlparse(img_url)
            if parsed.scheme and parsed.netloc:
                cleaned_images.append(img_url)

        logger.info(f"Found {len(cleaned_images)} images for lot")
        return cleaned_images[:5]  # Return max 5 images

    def extract_auction_id(self, url):
        """Extract auction ID from URL"""
        patterns = [
            r'/auction/(\d+)',
            r'/catalog/(\d+)',
            r'/company/(\d+)',
            r'auction_id=(\d+)',
            r'/auctions/(\d+)',
            r'/sale/(\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def scrape_catalog_page_enhanced(self, catalog_url):
        """Enhanced catalog page scraping with improved image extraction"""
        logger.info(f"Scraping catalog: {catalog_url}")
        content = self.get_page_content(catalog_url)
        if not content:
            logger.warning(f"Could not fetch content from {catalog_url}")
            return []

        soup = BeautifulSoup(content, 'html.parser')
        items = []

        # Extract auction info
        auction_id = self.extract_auction_id(catalog_url)
        auction_title = ""
        company_name = ""

        # Look for auction title in various places
        title_selectors = [
            'h1', 'h2.auction-title', '.title', '.auction-name',
            '[class*="title"]', '[class*="auction"]', 'title'
        ]

        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title_text = title_elem.get_text().strip()
                if len(title_text) > 5 and 'hibid' not in title_text.lower():
                    auction_title = title_text[:200]
                    break

        # Look for company name
        company_selectors = [
            '.company-name', '.auctioneer', '[class*="company"]',
            '[class*="auctioneer"]', '.seller'
        ]

        for selector in company_selectors:
            company_elem = soup.select_one(selector)
            if company_elem:
                company_text = company_elem.get_text().strip()
                if len(company_text) > 2:
                    company_name = company_text[:100]
                    break

        # Extract auction end time
        auction_end_time = None
        auction_time_remaining = None

        # Look for time information
        time_selectors = [
            '[class*="time"]', '[class*="end"]', '[class*="closing"]',
            '.auction-time', '.end-time', '.closing-time'
        ]

        for selector in time_selectors:
            time_elem = soup.select_one(selector)
            if time_elem:
                time_text = time_elem.get_text()
                end_time, time_remaining = self.parse_time_remaining(time_text)
                if end_time:
                    auction_end_time = end_time
                    auction_time_remaining = time_remaining
                    break

        # Look for lot containers with multiple strategies
        lot_containers = []

        # Strategy 1: Look for elements with lot/item in class or id
        lot_selectors = [
            '[class*="lot"]', '[id*="lot"]',
            '[class*="item"]', '[id*="item"]',
            'tr[class*="auction"]', 'div[class*="auction"]',
            '.product', '.listing'
        ]

        for selector in lot_selectors:
            elements = soup.select(selector)
            if elements:
                lot_containers.extend(elements)
                logger.info(f"Found {len(elements)} elements with selector: {selector}")

        # Strategy 2: Look for table rows that might contain lots
        table_rows = soup.select('tr')
        for row in table_rows:
            row_text = row.get_text().lower()
            if any(keyword in row_text for keyword in ['lot', 'item', 'bid', '$']):
                lot_containers.append(row)

        # Remove duplicates while preserving order
        seen = set()
        unique_containers = []
        for container in lot_containers:
            container_id = id(container)
            if container_id not in seen:
                seen.add(container_id)
                unique_containers.append(container)

        logger.info(f"Processing {len(unique_containers)} potential lot containers")

        for i, lot_elem in enumerate(unique_containers):
            try:
                lot_text = lot_elem.get_text()

                # Skip elements that are too short or don't look like lots
                if len(lot_text.strip()) < 10:
                    continue

                # Extract lot number with more flexible patterns
                lot_patterns = [
                    r'(?:Lot|Item|#)\s*[:\-]?\s*(\d+[a-zA-Z]?)',
                    r'(?:^|\s)(\d+[a-zA-Z]?)[:\-]',  # Number at start of line or after space
                    r'#(\d+[a-zA-Z]?)',
                    r'(\d{1,4}[a-zA-Z]?)\s*(?:\.|:|\-)',  # Number followed by punctuation
                ]

                lot_number = None
                for pattern in lot_patterns:
                    lot_match = re.search(pattern, lot_text, re.IGNORECASE)
                    if lot_match:
                        potential_lot = lot_match.group(1)
                        # Validate lot number (should be reasonable)
                        if potential_lot.isdigit() and 1 <= int(potential_lot) <= 9999:
                            lot_number = potential_lot
                            break
                        elif len(potential_lot) <= 6:  # Allow alphanumeric lot numbers
                            lot_number = potential_lot
                            break

                if not lot_number:
                    continue

                # Extract description with multiple strategies
                description = "No description"

                # Strategy 1: Look for specific description elements
                desc_selectors = [
                    '.description', '.title', '.name', '.item-title',
                    'h1', 'h2', 'h3', 'h4', 'h5', 'strong', 'b'
                ]

                for selector in desc_selectors:
                    desc_elem = lot_elem.select_one(selector)
                    if desc_elem:
                        desc_text = desc_elem.get_text().strip()
                        if len(desc_text) > 5 and not re.match(r'^(Lot|#|\d+)', desc_text):
                            description = desc_text[:300]
                            break

                # Strategy 2: Extract from text content
                if description == "No description":
                    text_parts = [part.strip() for part in lot_text.split('\n') if part.strip()]
                    for part in text_parts:
                        # Skip short parts, lot numbers, prices, and bid info
                        if (len(part) > 15 and
                            not re.match(r'^(Lot|#|\d+)', part) and
                            '$' not in part and
                            'bid' not in part.lower()):
                            description = part[:300]
                            break

                # Extract price information
                current_price = None
                price_text = ""

                price_patterns = [
                    r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                    r'USD\s*(\d+(?:\.\d{2})?)',
                    r'Current[:\s]*\$?(\d+(?:\.\d{2})?)',
                    r'Bid[:\s]*\$?(\d+(?:\.\d{2})?)',
                    r'Price[:\s]*\$?(\d+(?:\.\d{2})?)'
                ]

                for pattern in price_patterns:
                    price_match = re.search(pattern, lot_text)
                    if price_match:
                        price_text = price_match.group(0)
                        current_price = self.parse_price(price_text)
                        if current_price and current_price > 0:
                            break

                # Extract bid count
                bid_count = 0
                bid_patterns = [
                    r'(\d+)\s*Bids?',
                    r'Bids?[:\s]*(\d+)',
                    r'(\d+)\s*(?:bidders?|bids?)'
                ]

                for pattern in bid_patterns:
                    bid_match = re.search(pattern, lot_text, re.IGNORECASE)
                    if bid_match:
                        try:
                            bid_count = int(bid_match.group(1))
                            break
                        except ValueError:
                            continue

                # Extract individual item time (if different from auction)
                item_end_time = auction_end_time
                item_time_remaining = auction_time_remaining

                # Look for time info within the lot element
                time_elem = lot_elem.find(text=re.compile(r'End|Time.*Left|Closing', re.IGNORECASE))
                if time_elem and time_elem.parent:
                    time_text = time_elem.parent.get_text()
                    end_time, time_remaining = self.parse_time_remaining(time_text)
                    if end_time:
                        item_end_time = end_time
                        item_time_remaining = time_remaining

                # Extract images - this is the key improvement
                image_urls = self.extract_images(lot_elem, catalog_url)

                # If no images found in lot element, try to find images by lot number
                if not image_urls:
                    # Look for images with lot number in src, alt, or nearby text
                    all_images = soup.find_all('img')
                    for img in all_images:
                        img_src = img.get('src', '')
                        img_alt = img.get('alt', '')
                        img_class = ' '.join(img.get('class', []))

                        # Check if image is related to this lot
                        if (lot_number in img_src or
                            lot_number in img_alt or
                            lot_number in img_class):
                            full_url = urljoin(catalog_url, img_src)
                            if self.is_valid_image_url(full_url):
                                image_urls.append(full_url)

                item = AuctionItem(
                    lot_number=lot_number,
                    description=description,
                    current_price=current_price,
                    price_text=price_text,
                    bid_count=bid_count,
                    source=catalog_url,
                    auction_id=auction_id,
                    end_time=item_end_time,
                    time_remaining=item_time_remaining,
                    image_urls=image_urls,
                    auction_title=auction_title,
                    company_name=company_name
                )

                items.append(item)
                logger.info(f"Extracted lot {lot_number}: {description[:50]}... (${current_price if current_price else 'N/A'}) - {len(image_urls)} images")

            except Exception as e:
                logger.error(f"Error parsing lot {i+1}: {e}")
                continue

        logger.info(f"Successfully extracted {len(items)} items from {catalog_url}")
        return items

    def scrape_and_store_all(self, include_individual_auctions=True, max_auctions=5):
        """Main method to scrape and store all items in database"""
        logger.info(f"Starting enhanced HiBid scraper for {'zip code ' + self.zip_code if self.zip_code else 'Indiana'}...")

        all_items = []
        all_auctions = []

        # Scrape zip code specific auctions if zip code provided
        if self.zip_code:
            zip_auctions = self.scrape_zip_code_auctions_enhanced(max_auctions * 2)
            all_auctions.extend(zip_auctions)
            logger.info(f"Found {len(zip_auctions)} auctions near zip code {self.zip_code}")

        # Scrape individual auction pages
        if include_individual_auctions:
            auction_items = self.scrape_individual_auctions_enhanced(max_auctions)
            all_items.extend(auction_items)
            logger.info(f"Found {len(auction_items)} items from individual auctions")

        # Store in database
        stored_auctions = 0
        stored_items = 0

        for auction in all_auctions:
            try:
                self.db.save_auction(auction)
                stored_auctions += 1
            except Exception as e:
                logger.error(f"Error storing auction: {e}")

        for item in all_items:
            try:
                self.db.save_item(item)
                stored_items += 1
            except Exception as e:
                logger.error(f"Error storing item: {e}")

        logger.info(f"Stored {stored_auctions} auctions and {stored_items} items in database")
        return all_items, all_auctions

    def scrape_zip_code_auctions_enhanced(self, max_auctions=10):
        """Enhanced zip code auction scraping"""
        if not self.zip_code:
            return []

        content = self.get_page_content(self.zip_url)
        if not content:
            return []

        soup = BeautifulSoup(content, 'html.parser')
        auctions = []

        company_links = soup.find_all('a', href=re.compile(r'/company/\d+/'))

        for link in company_links[:max_auctions]:
            try:
                company_name = link.get_text().strip()
                company_url = urljoin(self.base_url, link['href'])
                auction_id = self.extract_auction_id(company_url)

                parent = link.parent
                while parent and parent.name != 'body':
                    parent_text = parent.get_text()

                    # Extract auction dates
                    date_match = re.search(r'Date\(s\)\s+([\d/\-\s]+)', parent_text)
                    dates = date_match.group(1).strip() if date_match else "Unknown"

                    # Extract location
                    map_link = parent.find('a', href=re.compile(r'google\.com/maps'))
                    location = "Unknown"
                    if map_link and 'query=' in map_link['href']:
                        location_query = map_link['href'].split('query=')[1]
                        location = location_query.replace('%2C', ',').replace('%20', ' ')[:100]

                    # Extract auction title
                    title_elem = parent.find(['h2', 'h3', 'h4'])
                    auction_title = title_elem.get_text().strip() if title_elem else company_name

                    # Extract time information
                    end_time, time_remaining = None, None
                    time_text = parent.find(text=re.compile(r'End|Closing|Time.*Left', re.IGNORECASE))
                    if time_text:
                        end_time, time_remaining = self.parse_time_remaining(time_text.parent.get_text())

                    # Extract bidding notice
                    bidding_notice = ""
                    notice_elem = parent.find(text=re.compile(r'Bidding Notice:|Auction Notice:'))
                    if notice_elem:
                        notice_parent = notice_elem.parent
                        if notice_parent:
                            bidding_notice = notice_parent.get_text().strip()[:200]

                    auction = AuctionInfo(
                        company_name=company_name,
                        company_url=company_url,
                        auction_title=auction_title,
                        dates=dates,
                        location=location,
                        bidding_notice=bidding_notice,
                        zip_code=self.zip_code,
                        end_time=end_time,
                        time_remaining=time_remaining,
                        auction_id=auction_id
                    )

                    auctions.append(auction)
                    break

            except Exception as e:
                logger.error(f"Error parsing auction info: {e}")
                continue

        return auctions

    def scrape_individual_auctions_enhanced(self, max_auctions=5):
        """Enhanced individual auction scraping"""
        # Use zip code URL if available, otherwise use main Indiana URL
        if self.zip_code:
            content = self.get_page_content(self.zip_url)
        else:
            content = self.get_page_content(self.indiana_url)

        if not content:
            return []

        soup = BeautifulSoup(content, 'html.parser')
        auction_links = []

        # Find auction links with improved patterns
        link_patterns = [
            r'/catalog/\d+',
            r'/auction/\d+',
            r'/auctions/\d+',
            r'/sale/\d+'
        ]

        for link in soup.find_all('a', href=True):
            href = link['href']
            for pattern in link_patterns:
                if re.search(pattern, href):
                    full_url = urljoin(self.base_url, href)
                    auction_links.append(full_url)
                    break

        # Remove duplicates and limit
        auction_links = list(set(auction_links))[:max_auctions]

        all_items = []
        for auction_url in auction_links:
            logger.info(f"Scraping auction: {auction_url}")
            items = self.scrape_catalog_page_enhanced(auction_url)
            all_items.extend(items)
            time.sleep(2)  # Be more respectful with delays

        return all_items

# Web Application Helper Functions
def get_recent_auctions(db_path='hibid_auctions.db', zip_code=None, hours=24):
    """Get recent auctions from database"""
    db = DatabaseManager(db_path)
    return db.get_active_auctions(zip_code)

def get_auction_items(auction_id, db_path='hibid_auctions.db'):
    """Get all items for a specific auction"""
    db = DatabaseManager(db_path)
    return db.get_items_by_auction(auction_id)

def search_items(query, db_path='hibid_auctions.db', limit=50):
    """Search for items by description"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT i.*, GROUP_CONCAT(img.image_url) as image_urls
            FROM items i
            LEFT JOIN item_images img ON i.id = img.item_id
            WHERE i.description LIKE ?
            AND datetime(i.scraped_at) > datetime('now', '-24 hours')
            GROUP BY i.id
            ORDER BY i.current_price DESC
            LIMIT ?
        ''', (f'%{query}%', limit))

        columns = [desc[0] for desc in cursor.description]
        items = []
        for row in cursor.fetchall():
            item = dict(zip(columns, row))
            if item['image_urls']:
                item['image_urls'] = item['image_urls'].split(',')
            else:
                item['image_urls'] = []
            items.append(item)
        return items

def validate_images(db_path='hibid_auctions.db'):
    """Validate and test image URLs in database"""
    logger.info("Validating image URLs...")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT image_url FROM item_images LIMIT 10')

        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        for (url,) in cursor.fetchall():
            try:
                response = session.head(url, timeout=5)
                status = response.status_code
                content_type = response.headers.get('content-type', '')

                logger.info(f"URL: {url}")
                logger.info(f"  Status: {status}")
                logger.info(f"  Content-Type: {content_type}")
                logger.info(f"  Valid: {'Yes' if status == 200 and 'image' in content_type else 'No'}")
                logger.info("-" * 50)

            except Exception as e:
                logger.error(f"Error checking {url}: {e}")
