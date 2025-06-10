#!/usr/bin/env python3
"""
Configuration file for the Auction Scraper Application
Handles environment-specific settings and constants
"""

import os
from pathlib import Path
from typing import Optional

class AppConfig:
    """Application configuration class"""
    
    # Environment detection
    IS_HF_SPACE = os.getenv('SPACE_ID') is not None
    IS_LOCAL_DEV = not IS_HF_SPACE
    
    # Database configuration
    DB_NAME = 'auction_data.db'
    DATA_DIR = Path('data')
    DB_PATH = DATA_DIR / DB_NAME if IS_HF_SPACE else DB_NAME
    
    # Scraping configuration - Conservative settings
    SCRAPE_INTERVAL_HOURS = 6  # How often to auto-scrape
    MAX_AUCTIONS_PER_RUN = 3   # Limit per scraping session
    MAX_ITEMS_DISPLAY = 100    # Max items to show at once
    REQUEST_DELAY = 2          # Seconds between requests
    MAX_RETRIES = 3            # Max retries for failed requests
    REQUEST_TIMEOUT = 15       # Request timeout in seconds
    
    # Default settings
    DEFAULT_ZIP_CODE = "46032"  # Indianapolis area
    DATA_RETENTION_HOURS = 24   # How long to keep data
    
    # UI Configuration
    TABLE_HEIGHT = 600
    AUTO_REFRESH_SECONDS = 30
    
    # Gradio launch settings
    GRADIO_CONFIG = {
        "share": False,
        "server_name": "0.0.0.0" if IS_HF_SPACE else "127.0.0.1",
        "server_port": 7860,
        "show_error": True,
        "favicon_path": None,
    }
    
    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'auction_app.log'
    
    # Auction site configurations
    AUCTION_SITES = {
        'earls': {
            'name': "Earl's Auction Company",
            'base_url': "https://www.earlsauction.com",
            'enabled': True,
            'max_auctions': 2,  # Conservative limit
        },
        'hibid': {
            'name': "HiBid Auctions",
            'base_url': "https://hibid.com",
            'enabled': True,
            'max_auctions': 2,  # Conservative limit
        }
    }
    
    # Price validation settings
    MIN_VALID_PRICE = 0.01
    MAX_VALID_PRICE = 50000.00
    
    # Image handling
    MAX_IMAGES_PER_ITEM = 5
    MIN_IMAGE_SIZE = 50  # pixels
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        if cls.IS_HF_SPACE:
            cls.DATA_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_user_agent(cls) -> str:
        """Get user agent string for requests"""
        return 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    @classmethod
    def get_request_headers(cls) -> dict:
        """Get default request headers"""
        return {
            'User-Agent': cls.get_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

# Initialize directories on import
AppConfig.setup_directories()