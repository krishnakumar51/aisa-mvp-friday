# COMPLETELY FIXED Mobile Setup Template - NO JSON ERRORS!
MOBILE_SETUP_TEMPLATE = '''# NOTE: TouchAction and MultiAction are deprecated in modern Appium clients.
# DO NOT import or use TouchAction or MultiAction in generated code.
# Prefer Appium 'mobile:' execute_script gestures (mobile: longClickGesture, mobile: clickGesture, mobile: swipeGesture)
# or the W3C Actions API (PointerInput/Sequence) for advanced gestures.

import os
import sys
import time
import logging
import subprocess
from typing import Optional, Dict, Tuple, List, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging.handlers
from pathlib import Path

# Third-party imports
from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException,
    InvalidElementStateException,
    ElementNotInteractableException
)

# HARDCODED DEVICE CONFIGURATION (NEVER CHANGE)
DEFAULT_APP_PACKAGE = "com.microsoft.office.outlook"
DEFAULT_APP_ACTIVITY = ".MainActivity"
DEFAULT_DEVICE_UDID = "ZD222GXYPV"
DEFAULT_DEVICE_NAME = "ZD222GXYPV"
DEFAULT_APPIUM_SERVER = "http://localhost:4723"

# These are intentionally hardcoded for production stability
# DO NOT parameterize these values - they are device-specific constants
DEVICE_CONFIG = {
    "platformName": "Android",
    "platformVersion": "14.0",
    "deviceName": DEFAULT_DEVICE_NAME,
    "udid": DEFAULT_DEVICE_UDID,
    "automationName": "UiAutomator2",
    "appPackage": DEFAULT_APP_PACKAGE,
    "appActivity": DEFAULT_APP_ACTIVITY,
    "noReset": True,
    "fullReset": False,
    "newCommandTimeout": 300,
    "androidKeepAppDataOnInstall": True,
    "skipDeviceInitialization": False,
    "skipServerInstallation": True,
    "ignoreHiddenApiPolicyError": True,
    "disableIdLocatorAutocompletion": True,
}

class AutomationError(Exception):
    """Base exception for automation failures"""
    pass

class ElementNotFoundError(AutomationError):
    """Element could not be located despite multiple strategies"""
    pass

class ElementInteractionError(AutomationError):
    """Element found but interaction failed"""
    pass

class GestureExecutionError(AutomationError):
    """Gesture execution failed"""
    pass

@dataclass
class ElementStrategy:
    """Element finding strategy with multiple fallback methods"""
    primary: str
    fallbacks: List[str]
    timeout: int = 10
    description: str = ""

# ======================== ENHANCED HELPER FUNCTIONS ========================

def setup_driver() -> webdriver.Remote:
    """
    Initialize Appium driver with production-grade configuration.
    Uses hardcoded device settings for consistency and reliability.
    """
    try:
        options = UiAutomator2Options()
        options.load_capabilities(DEVICE_CONFIG)
        
        driver = webdriver.Remote(DEFAULT_APPIUM_SERVER, options=options)
        driver.implicitly_wait(5)
        
        # Verify device connection and app state
        if not driver.is_app_installed(DEFAULT_APP_PACKAGE):
            raise AutomationError(f"App {DEFAULT_APP_PACKAGE} not installed on device")
        
        return driver
    except Exception as e:
        raise AutomationError(f"Driver initialization failed: {str(e)}")

def find_element_smart(driver: webdriver.Remote, 
                      element_id: str = None,
                      xpath: str = None, 
                      class_name: str = None,
                      text: str = None,
                      accessibility_id: str = None,
                      content_desc: str = None,
                      resource_id: str = None,
                      timeout: int = 15) -> webdriver.WebElement:
    """
    Universal element finder with intelligent fallback strategies.
    Tries multiple locator strategies in order of reliability.
    
    Args:
        driver: Appium WebDriver instance
        element_id: Element ID (highest priority)
        xpath: XPath selector
        class_name: Android class name
        text: Visible text content
        accessibility_id: Accessibility identifier
        content_desc: Content description
        resource_id: Android resource ID
        timeout: Maximum wait time in seconds
    
    Returns:
        WebElement: Found element
        
    Raises:
        ElementNotFoundError: If element cannot be found with any strategy
    """
    wait = WebDriverWait(driver, timeout)
    strategies = []
    
    # Build strategy list in order of reliability
    if element_id:
        strategies.append((AppiumBy.ID, element_id, "ID"))
    if accessibility_id:
        strategies.append((AppiumBy.ACCESSIBILITY_ID, accessibility_id, "ACCESSIBILITY_ID"))
    if resource_id:
        strategies.append((AppiumBy.ID, resource_id, "RESOURCE_ID"))
    if xpath:
        strategies.append((AppiumBy.XPATH, xpath, "XPATH"))
    if class_name:
        strategies.append((AppiumBy.CLASS_NAME, class_name, "CLASS_NAME"))
    if text:
        # Try both exact text and partial text
        strategies.append((AppiumBy.XPATH, f'//*[@text="{text}"]', "EXACT_TEXT"))
        strategies.append((AppiumBy.XPATH, f'//*[contains(@text, "{text}")]', "PARTIAL_TEXT"))
    if content_desc:
        strategies.append((AppiumBy.XPATH, f'//*[@content-desc="{content_desc}"]', "CONTENT_DESC"))
    
    last_error = None
    
    # Try each strategy
    for by, value, strategy_name in strategies:
        try:
            element = wait.until(EC.presence_of_element_located((by, value)))
            if element and element.is_displayed():
                return element
        except (TimeoutException, NoSuchElementException, StaleElementReferenceException) as e:
            last_error = e
            continue
    
    # If all strategies fail, try coordinate-based fallback (screen center)
    try:
        screen_size = driver.get_window_size()
        center_x = screen_size["width"] // 2
        center_y = screen_size["height"] // 2
        
        # Look for any clickable element near screen center
        element = driver.find_element(
            AppiumBy.XPATH, 
            f'//*[@clickable="true" and @bounds[contains(., "{center_x-50},{center_y-50}")]]'
        )
        if element:
            return element
    except Exception:
        pass
    
    # Final fallback: OCR-based text detection (if text provided)
    if text:
        try:
            screenshot = driver.get_screenshot_as_png()
            # This would require OCR integration - placeholder for now
            # element = find_element_by_ocr(screenshot, text)
            pass
        except Exception:
            pass
    
    raise ElementNotFoundError(f"Element not found with any strategy. Last error: {last_error}")

def click_smart(driver: webdriver.Remote,
               element: webdriver.WebElement = None,
               x: int = None, 
               y: int = None,
               retry_count: int = 3) -> bool:
    """
    Universal click function with multiple fallback strategies.
    
    Args:
        driver: Appium WebDriver instance
        element: Target element (if available)
        x, y: Coordinate-based click (fallback)
        retry_count: Number of retry attempts
    
    Returns:
        bool: True if click succeeded, False otherwise
    """
    for attempt in range(retry_count):
        try:
            if element:
                # Strategy 1: Standard element click
                try:
                    if element.is_displayed() and element.is_enabled():
                        element.click()
                        time.sleep(0.5)  # Brief pause after click
                        return True
                except (ElementNotInteractableException, StaleElementReferenceException):
                    # Element became stale, try to relocate
                    pass
                
                # Strategy 2: Element coordinate-based click
                try:
                    location = element.location
                    size = element.size
                    center_x = location['x'] + size['width'] // 2
                    center_y = location['y'] + size['height'] // 2
                    
                    driver.execute_script("mobile: clickGesture", {
                        "x": center_x,
                        "y": center_y
                    })
                    time.sleep(0.5)
                    return True
                except Exception:
                    pass
            
            # Strategy 3: Direct coordinate click
            if x is not None and y is not None:
                try:
                    driver.execute_script("mobile: clickGesture", {
                        "x": x,
                        "y": y
                    })
                    time.sleep(0.5)
                    return True
                except Exception:
                    pass
            
            # Strategy 4: ADB tap fallback
            if x is not None and y is not None:
                try:
                    subprocess.run([
                        "adb", "-s", DEFAULT_DEVICE_UDID, 
                        "shell", "input", "tap", str(x), str(y)
                    ], check=True, timeout=5)
                    time.sleep(0.5)
                    return True
                except Exception:
                    pass
            
            time.sleep(1)  # Brief pause before retry
            
        except Exception as e:
            if attempt == retry_count - 1:
                raise ElementInteractionError(f"Click failed after {retry_count} attempts: {str(e)}")
            time.sleep(1)
    
    return False

def type_smart(driver: webdriver.Remote,
              element: webdriver.WebElement,
              text: str,
              clear_method: str = "select_all",
              retry_count: int = 3) -> bool:
    """
    Universal text input with smart clearing strategies.
    
    Args:
        driver: Appium WebDriver instance
        element: Target input element
        text: Text to input
        clear_method: Clearing strategy ('select_all', 'backspace', 'clear')
        retry_count: Number of retry attempts
    
    Returns:
        bool: True if input succeeded, False otherwise
    """
    for attempt in range(retry_count):
        try:
            # Ensure element is interactable
            if not element.is_displayed() or not element.is_enabled():
                time.sleep(1)
                continue
            
            # Focus on element first
            element.click()
            time.sleep(0.3)
            
            # Clear existing content based on strategy
            if clear_method == "select_all":
                try:
                    # Select all and delete (most reliable)
                    driver.execute_script("mobile: key", {"key": 97})  # Ctrl+A equivalent
                    time.sleep(0.2)
                    element.clear()
                    time.sleep(0.2)
                except Exception:
                    pass
            
            elif clear_method == "backspace":
                try:
                    # Get current text length and backspace
                    current_text = element.get_attribute("text") or ""
                    for _ in range(len(current_text) + 5):  # Extra backspaces for safety
                        driver.execute_script("mobile: key", {"key": 67})  # Backspace
                        time.sleep(0.05)
                    time.sleep(0.3)
                except Exception:
                    pass
            
            elif clear_method == "clear":
                try:
                    element.clear()
                    time.sleep(0.2)
                except Exception:
                    pass
            
            # Input new text
            element.send_keys(text)
            time.sleep(0.5)
            
            # Verify text was entered correctly
            try:
                entered_text = element.get_attribute("text") or ""
                if text.lower() in entered_text.lower():
                    return True
            except Exception:
                pass
            
            # Hide keyboard if present
            try:
                if driver.is_keyboard_shown():
                    driver.hide_keyboard()
            except Exception:
                pass
            
            return True
            
        except Exception as e:
            if attempt == retry_count - 1:
                raise ElementInteractionError(f"Text input failed after {retry_count} attempts: {str(e)}")
            time.sleep(1)
    
    return False

def long_press_smart(driver: webdriver.Remote,
                    element: webdriver.WebElement = None,
                    x: int = None,
                    y: int = None, 
                    duration: int = 15000,
                    retry_count: int = 3) -> bool:
    """
    PRODUCTION-GRADE long press for Microsoft CAPTCHA and other scenarios.
    Uses modern Appium mobile: gestures with ADB fallback.
    
    Args:
        driver: Appium WebDriver instance
        element: Target element (preferred)
        x, y: Coordinate-based long press (fallback)
        duration: Hold duration in milliseconds (15000ms for Microsoft CAPTCHA)
        retry_count: Number of retry attempts
    
    Returns:
        bool: True if long press succeeded, False otherwise
    """
    for attempt in range(retry_count):
        try:
            target_x, target_y = x, y
            
            # Get coordinates from element if provided
            if element:
                try:
                    location = element.location
                    size = element.size
                    target_x = location['x'] + size['width'] // 2
                    target_y = location['y'] + size['height'] // 2
                except Exception:
                    pass
            
            # Strategy 1: Modern Appium mobile:longClickGesture (PREFERRED)
            if target_x is not None and target_y is not None:
                try:
                    driver.execute_script("mobile: longClickGesture", {
                        "x": target_x,
                        "y": target_y,
                        "duration": duration
                    })
                    time.sleep(1)  # Brief pause after gesture
                    return True
                except Exception as e:
                    pass
            
            # Strategy 2: Element-based long press (if element available)
            if element:
                try:
                    driver.execute_script("mobile: longClickGesture", {
                        "elementId": element.id,
                        "duration": duration
                    })
                    time.sleep(1)
                    return True
                except Exception:
                    pass
            
            # Strategy 3: ADB shell command fallback (MOST RELIABLE)
            if target_x is not None and target_y is not None:
                try:
                    # Convert duration to seconds for ADB
                    duration_seconds = duration / 1000.0
                    
                    # Use ADB shell input swipe with same start/end coordinates for long press
                    subprocess.run([
                        "adb", "-s", DEFAULT_DEVICE_UDID, "shell", "input", "swipe", 
                        str(target_x), str(target_y), str(target_x), str(target_y), 
                        str(int(duration))
                    ], check=True, timeout=duration_seconds + 5)
                    
                    time.sleep(1)
                    return True
                except Exception:
                    pass
            
            # Strategy 4: Screen center fallback for CAPTCHA scenarios
            if target_x is None or target_y is None:
                try:
                    screen_size = driver.get_window_size()
                    center_x = screen_size["width"] // 2  
                    center_y = screen_size["height"] // 2
                    
                    driver.execute_script("mobile: longClickGesture", {
                        "x": center_x,
                        "y": center_y,
                        "duration": duration
                    })
                    time.sleep(1)
                    return True
                except Exception:
                    pass
            
            time.sleep(1)  # Brief pause before retry
            
        except Exception as e:
            if attempt == retry_count - 1:
                raise GestureExecutionError(f"Long press failed after {retry_count} attempts: {str(e)}")
            time.sleep(1)
    
    return False

def wait_and_find(driver: webdriver.Remote,
                 timeout: int = 15,
                 **element_params) -> Optional[webdriver.WebElement]:
    """
    Wait for element with smart finding strategies.
    
    Args:
        driver: Appium WebDriver instance
        timeout: Maximum wait time
        **element_params: Element identification parameters
    
    Returns:
        WebElement if found, None otherwise
    """
    try:
        return find_element_smart(driver, timeout=timeout, **element_params)
    except ElementNotFoundError:
        return None

def swipe_smart(driver: webdriver.Remote,
               direction: str = "down",
               distance: float = 0.5,
               speed: int = 1000) -> bool:
    """
    Universal swipe gesture with configurable parameters.
    
    Args:
        driver: Appium WebDriver instance
        direction: Swipe direction ('up', 'down', 'left', 'right')
        distance: Swipe distance as fraction of screen (0.0-1.0)
        speed: Swipe duration in milliseconds
    
    Returns:
        bool: True if swipe succeeded
    """
    try:
        screen_size = driver.get_window_size()
        screen_width = screen_size["width"]
        screen_height = screen_size["height"]
        
        # Calculate swipe coordinates
        center_x = screen_width // 2
        center_y = screen_height // 2
        
        if direction == "up":
            start_x, start_y = center_x, int(center_y + (screen_height * distance / 2))
            end_x, end_y = center_x, int(center_y - (screen_height * distance / 2))
        elif direction == "down":
            start_x, start_y = center_x, int(center_y - (screen_height * distance / 2))
            end_x, end_y = center_x, int(center_y + (screen_height * distance / 2))
        elif direction == "left":
            start_x, start_y = int(center_x + (screen_width * distance / 2)), center_y
            end_x, end_y = int(center_x - (screen_width * distance / 2)), center_y
        elif direction == "right":
            start_x, start_y = int(center_x - (screen_width * distance / 2)), center_y
            end_x, end_y = int(center_x + (screen_width * distance / 2)), center_y
        else:
            raise ValueError(f"Invalid direction: {direction}")
        
        # Execute swipe using modern mobile: gesture
        driver.execute_script("mobile: swipeGesture", {
            "startX": start_x,
            "startY": start_y,
            "endX": end_x,
            "endY": end_y,
            "duration": speed
        })
        
        time.sleep(0.5)  # Brief pause after swipe
        return True
        
    except Exception as e:
        raise GestureExecutionError(f"Swipe failed: {str(e)}")

def validate_element_state(element: webdriver.WebElement,
                          should_be_displayed: bool = True,
                          should_be_enabled: bool = True,
                          should_contain_text: str = None) -> bool:
    """
    Validate element state against expected conditions.
    
    Args:
        element: Target element
        should_be_displayed: Expected display state
        should_be_enabled: Expected enabled state  
        should_contain_text: Expected text content (partial match)
    
    Returns:
        bool: True if all validations pass
    """
    try:
        if should_be_displayed and not element.is_displayed():
            return False
            
        if should_be_enabled and not element.is_enabled():
            return False
            
        if should_contain_text:
            element_text = element.get_attribute("text") or ""
            if should_contain_text.lower() not in element_text.lower():
                return False
        
        return True
    except Exception:
        return False

def handle_popup_smart(driver: webdriver.Remote,
                      accept_text: List[str] = ["OK", "Allow", "Accept", "Continue"],
                      dismiss_text: List[str] = ["Cancel", "Dismiss", "No", "Skip"]) -> bool:
    """
    Smart popup handling with configurable accept/dismiss text patterns.
    
    Args:
        driver: Appium WebDriver instance
        accept_text: Text patterns for accept buttons
        dismiss_text: Text patterns for dismiss buttons
    
    Returns:
        bool: True if popup was handled
    """
    try:
        # Look for popup elements
        popup_indicators = [
            '//android.widget.Dialog',
            '//*[contains(@class, "alert")]',
            '//*[contains(@class, "popup")]',
            '//*[contains(@class, "modal")]'
        ]
        
        for indicator in popup_indicators:
            try:
                popup = driver.find_element(AppiumBy.XPATH, indicator)
                if popup and popup.is_displayed():
                    
                    # Try to find accept button first
                    for accept_term in accept_text:
                        try:
                            accept_btn = driver.find_element(
                                AppiumBy.XPATH, 
                                f'//*[@text="{accept_term}" or contains(@text, "{accept_term}")]'
                            )
                            if accept_btn and accept_btn.is_displayed():
                                click_smart(driver, accept_btn)
                                time.sleep(1)
                                return True
                        except Exception:
                            continue
                    
                    # If no accept button, try dismiss
                    for dismiss_term in dismiss_text:
                        try:
                            dismiss_btn = driver.find_element(
                                AppiumBy.XPATH,
                                f'//*[@text="{dismiss_term}" or contains(@text, "{dismiss_term}")]'
                            )
                            if dismiss_btn and dismiss_btn.is_displayed():
                                click_smart(driver, dismiss_btn)
                                time.sleep(1)
                                return True
                        except Exception:
                            continue
                            
            except Exception:
                continue
        
        return False
    except Exception:
        return False

def capture_screenshot_smart(driver: webdriver.Remote,
                           filename: str = None,
                           directory: str = "./screenshots") -> str:
    """
    Capture screenshot with automatic naming and directory creation.
    
    Args:
        driver: Appium WebDriver instance
        filename: Custom filename (optional)
        directory: Screenshot directory
    
    Returns:
        str: Path to saved screenshot
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            timestamp = int(time.time())
            filename = f"screenshot_{timestamp}.png"
        
        if not filename.endswith('.png'):
            filename += '.png'
        
        filepath = os.path.join(directory, filename)
        
        # Capture and save screenshot
        screenshot = driver.get_screenshot_as_png()
        with open(filepath, 'wb') as f:
            f.write(screenshot)
        
        return filepath
    except Exception as e:
        raise AutomationError(f"Screenshot capture failed: {str(e)}")

def dismiss_keyboard_smart(driver: webdriver.Remote) -> bool:
    """
    Smart keyboard dismissal with multiple strategies.
    
    Args:
        driver: Appium WebDriver instance
    
    Returns:
        bool: True if keyboard was dismissed or not present
    """
    try:
        if not driver.is_keyboard_shown():
            return True
        
        # Strategy 1: Standard hide keyboard
        try:
            driver.hide_keyboard()
            time.sleep(0.5)
            if not driver.is_keyboard_shown():
                return True
        except Exception:
            pass
        
        # Strategy 2: Back button press
        try:
            driver.back()
            time.sleep(0.5)
            if not driver.is_keyboard_shown():
                return True
        except Exception:
            pass
        
        # Strategy 3: Click outside keyboard area
        try:
            screen_size = driver.get_window_size()
            driver.execute_script("mobile: clickGesture", {
                "x": screen_size["width"] // 2,
                "y": 100  # Click near top of screen
            })
            time.sleep(0.5)
            if not driver.is_keyboard_shown():
                return True
        except Exception:
            pass
        
        return False
    except Exception:
        return False

def scroll_to_element_smart(driver: webdriver.Remote,
                           target_text: str = None,
                           target_element_params: Dict = None,
                           max_scrolls: int = 10,
                           scroll_direction: str = "down") -> Optional[webdriver.WebElement]:
    """
    Scroll until target element is found.
    
    Args:
        driver: Appium WebDriver instance
        target_text: Text to search for
        target_element_params: Element parameters for find_element_smart
        max_scrolls: Maximum scroll attempts
        scroll_direction: Scroll direction
    
    Returns:
        WebElement if found, None otherwise
    """
    for scroll_attempt in range(max_scrolls):
        try:
            # Try to find element first
            if target_element_params:
                element = wait_and_find(driver, timeout=2, **target_element_params)
                if element:
                    return element
            elif target_text:
                element = wait_and_find(driver, timeout=2, text=target_text)
                if element:
                    return element
            
            # Scroll and try again
            swipe_smart(driver, direction=scroll_direction, distance=0.6)
            time.sleep(1)
            
        except Exception:
            continue
    
    return None

# Driver cleanup function
def cleanup_driver(driver: webdriver.Remote):
    """
    Clean up driver resources safely.
    
    Args:
        driver: Appium WebDriver instance to clean up
    """
    try:
        if driver:
            driver.quit()
    except Exception:
        pass  # Ignore cleanup errors

'''

WEB_SETUP_TEMPLATE = '''import os
import sys
import time
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path

# Third-party imports  
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page, TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import expect

class WebAutomationError(Exception):
    """Base exception for web automation failures"""
    pass

class WebElementNotFoundError(WebAutomationError):
    """Web element could not be located despite multiple strategies"""
    pass

class WebElementInteractionError(WebAutomationError):
    """Web element found but interaction failed"""
    pass

# ======================== ENHANCED WEB HELPER FUNCTIONS ========================

def setup_browser(headless: bool = False, 
                 stealth: bool = True,
                 viewport: Dict[str, int] = None) -> tuple[Browser, BrowserContext, Page]:
    """
    Initialize Playwright browser with stealth configuration.
    
    Args:
        headless: Run browser in headless mode
        stealth: Enable stealth mode to avoid detection
        viewport: Custom viewport dimensions
    
    Returns:
        tuple: (browser, context, page) instances
    """
    try:
        playwright = sync_playwright().start()
        
        # Stealth browser configuration
        browser_args = [
            "--no-first-run",
            "--no-service-autorun", 
            "--no-default-browser-check",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-extensions",
            "--disable-default-apps",
            "--disable-translate",
            "--disable-sync",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            "--disable-client-side-phishing-detection",
            "--disable-component-extensions-with-background-pages",
            "--no-sandbox",
            "--disable-web-security",
            "--disable-features=TranslateUI",
            "--disable-ipc-flooding-protection"
        ]
        
        if stealth:
            browser_args.extend([
                "--disable-blink-features=AutomationControlled",
                "--disable-automation",
                "--disable-infobars"
            ])
        
        browser = playwright.chromium.launch(
            headless=headless,
            args=browser_args
        )
        
        # Context with stealth settings
        context_options = {
            "viewport": viewport or {"width": 1366, "height": 768},
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            "extra_http_headers": {
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8"
            }
        }
        
        context = browser.new_context(**context_options)
        
        if stealth:
            # Additional stealth measures
            context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                });
                
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });
                
                window.chrome = {
                    runtime: {},
                };
            """)
        
        page = context.new_page()
        return browser, context, page
        
    except Exception as e:
        raise WebAutomationError(f"Browser setup failed: {str(e)}")

def find_element_web_smart(page: Page,
                          selector: str = None,
                          text: str = None, 
                          xpath: str = None,
                          css: str = None,
                          role: str = None,
                          placeholder: str = None,
                          timeout: int = 15000) -> Any:
    """
    Universal web element finder with multiple strategies.
    
    Args:
        page: Playwright Page instance
        selector: CSS selector
        text: Element text content  
        xpath: XPath selector
        css: CSS selector (alternative)
        role: ARIA role
        placeholder: Input placeholder text
        timeout: Timeout in milliseconds
    
    Returns:
        Playwright Locator for the found element
        
    Raises:
        WebElementNotFoundError: If element cannot be found
    """
    try:
        # Try strategies in order of reliability
        if selector:
            locator = page.locator(selector)
            locator.wait_for(timeout=timeout, state="visible")
            return locator
            
        if text:
            # Try exact text first, then partial
            try:
                locator = page.get_by_text(text, exact=True)
                locator.wait_for(timeout=timeout//2, state="visible")
                return locator
            except:
                locator = page.get_by_text(text, exact=False)
                locator.wait_for(timeout=timeout//2, state="visible") 
                return locator
                
        if role:
            locator = page.get_by_role(role)
            locator.wait_for(timeout=timeout, state="visible")
            return locator
            
        if placeholder:
            locator = page.get_by_placeholder(placeholder)
            locator.wait_for(timeout=timeout, state="visible")
            return locator
            
        if xpath:
            locator = page.locator(f"xpath={xpath}")
            locator.wait_for(timeout=timeout, state="visible")
            return locator
            
        if css:
            locator = page.locator(css)
            locator.wait_for(timeout=timeout, state="visible")
            return locator
            
        raise WebElementNotFoundError("No valid selector provided")
        
    except PlaywrightTimeoutError:
        raise WebElementNotFoundError(f"Element not found within {timeout}ms")
    except Exception as e:
        raise WebElementNotFoundError(f"Element search failed: {str(e)}")

def click_web_smart(page: Page,
                   locator: Any = None,
                   selector: str = None,
                   text: str = None,
                   retry_count: int = 3,
                   force: bool = False) -> bool:
    """
    Universal web click with multiple strategies.
    
    Args:
        page: Playwright Page instance
        locator: Playwright locator (preferred)
        selector: CSS selector fallback
        text: Text-based selector fallback
        retry_count: Number of retry attempts
        force: Force click even if element not in viewport
    
    Returns:
        bool: True if click succeeded
    """
    for attempt in range(retry_count):
        try:
            target_locator = locator
            
            if not target_locator and selector:
                target_locator = page.locator(selector)
            elif not target_locator and text:
                target_locator = page.get_by_text(text)
            
            if not target_locator:
                raise WebElementInteractionError("No valid locator for click")
            
            # Wait for element to be clickable
            target_locator.wait_for(state="visible", timeout=10000)
            
            # Try standard click first
            try:
                target_locator.click(timeout=5000, force=force)
                time.sleep(0.5)
                return True
            except:
                # Try JavaScript click fallback
                page.evaluate("arguments[0].click()", target_locator.element_handle())
                time.sleep(0.5)
                return True
                
        except Exception as e:
            if attempt == retry_count - 1:
                raise WebElementInteractionError(f"Click failed after {retry_count} attempts: {str(e)}")
            time.sleep(1)
    
    return False

def type_web_smart(page: Page,
                  locator: Any = None,
                  selector: str = None,
                  text: str = "",
                  clear_first: bool = True,
                  retry_count: int = 3) -> bool:
    """
    Universal web text input with smart strategies.
    
    Args:
        page: Playwright Page instance  
        locator: Playwright locator (preferred)
        selector: CSS selector fallback
        text: Text to input
        clear_first: Clear field before typing
        retry_count: Number of retry attempts
    
    Returns:
        bool: True if input succeeded
    """
    for attempt in range(retry_count):
        try:
            target_locator = locator
            
            if not target_locator and selector:
                target_locator = page.locator(selector)
            
            if not target_locator:
                raise WebElementInteractionError("No valid locator for text input")
            
            # Wait for element
            target_locator.wait_for(state="visible", timeout=10000)
            
            # Clear if requested
            if clear_first:
                target_locator.clear()
                time.sleep(0.2)
            
            # Type text
            target_locator.fill(text)
            time.sleep(0.5)
            
            # Verify text was entered
            current_value = target_locator.input_value()
            if text in current_value:
                return True
                
        except Exception as e:
            if attempt == retry_count - 1:
                raise WebElementInteractionError(f"Text input failed after {retry_count} attempts: {str(e)}")
            time.sleep(1)
    
    return False

def wait_for_web_element(page: Page,
                        selector: str = None,
                        text: str = None,
                        state: str = "visible", 
                        timeout: int = 15000) -> bool:
    """
    Wait for web element with configurable state.
    
    Args:
        page: Playwright Page instance
        selector: Element selector
        text: Element text
        state: Expected state ('visible', 'hidden', 'attached', 'detached')
        timeout: Timeout in milliseconds
    
    Returns:
        bool: True if element reached expected state
    """
    try:
        if selector:
            locator = page.locator(selector)
        elif text:
            locator = page.get_by_text(text)
        else:
            return False
        
        locator.wait_for(state=state, timeout=timeout)
        return True
    except:
        return False

def handle_web_popup_smart(page: Page,
                          accept_patterns: List[str] = ["OK", "Accept", "Allow", "Continue"],
                          dismiss_patterns: List[str] = ["Cancel", "Dismiss", "No", "Skip"]) -> bool:
    """
    Handle web popups, dialogs, and alerts.
    
    Args:
        page: Playwright Page instance
        accept_patterns: Text patterns for accept buttons
        dismiss_patterns: Text patterns for dismiss buttons
    
    Returns:
        bool: True if popup was handled
    """
    try:
        # Handle JavaScript dialogs
        def handle_dialog(dialog):
            dialog.accept()
        
        page.on("dialog", handle_dialog)
        
        # Look for modal/popup elements
        popup_selectors = [
            '[role="dialog"]',
            '.modal',
            '.popup', 
            '.alert',
            '[aria-modal="true"]'
        ]
        
        for selector in popup_selectors:
            try:
                popup = page.locator(selector)
                if popup.is_visible():
                    
                    # Try accept buttons first
                    for pattern in accept_patterns:
                        try:
                            btn = popup.get_by_text(pattern, exact=False)
                            if btn.is_visible():
                                btn.click()
                                time.sleep(1)
                                return True
                        except:
                            continue
                    
                    # Try dismiss buttons
                    for pattern in dismiss_patterns:
                        try:
                            btn = popup.get_by_text(pattern, exact=False)
                            if btn.is_visible():
                                btn.click()
                                time.sleep(1)
                                return True
                        except:
                            continue
            except:
                continue
        
        return False
    except:
        return False

def capture_web_screenshot(page: Page,
                          filename: str = None,
                          directory: str = "./screenshots",
                          full_page: bool = True) -> str:
    """
    Capture web page screenshot.
    
    Args:
        page: Playwright Page instance
        filename: Custom filename
        directory: Screenshot directory  
        full_page: Capture full page or viewport only
    
    Returns:
        str: Path to saved screenshot
    """
    try:
        os.makedirs(directory, exist_ok=True)
        
        if not filename:
            timestamp = int(time.time())
            filename = f"web_screenshot_{timestamp}.png"
        
        if not filename.endswith('.png'):
            filename += '.png'
        
        filepath = os.path.join(directory, filename)
        
        page.screenshot(path=filepath, full_page=full_page)
        return filepath
    except Exception as e:
        raise WebAutomationError(f"Screenshot capture failed: {str(e)}")

def scroll_web_smart(page: Page,
                    direction: str = "down", 
                    pixels: int = None,
                    to_element: str = None) -> bool:
    """
    Smart web page scrolling.
    
    Args:
        page: Playwright Page instance
        direction: Scroll direction ('up', 'down', 'left', 'right')
        pixels: Pixels to scroll (optional)
        to_element: Scroll to specific element (optional)
    
    Returns:
        bool: True if scroll succeeded
    """
    try:
        if to_element:
            # Scroll to specific element
            page.locator(to_element).scroll_into_view_if_needed()
            return True
        
        if not pixels:
            # Default scroll distances
            pixels = 500 if direction in ["up", "down"] else 300
        
        # Determine scroll coordinates
        if direction == "down":
            page.evaluate(f"window.scrollBy(0, {pixels})")
        elif direction == "up":
            page.evaluate(f"window.scrollBy(0, -{pixels})")
        elif direction == "right":
            page.evaluate(f"window.scrollBy({pixels}, 0)")
        elif direction == "left":
            page.evaluate(f"window.scrollBy(-{pixels}, 0)")
        else:
            return False
        
        time.sleep(0.5)
        return True
    except:
        return False

# Browser cleanup function
def cleanup_browser(browser: Browser, context: BrowserContext = None, page: Page = None):
    """
    Clean up browser resources safely.
    
    Args:
        browser: Browser instance to clean up
        context: Context instance (optional)
        page: Page instance (optional)
    """
    try:
        if page:
            page.close()
        if context:
            context.close()
        if browser:
            browser.close()
    except Exception:
        pass  # Ignore cleanup errors

'''