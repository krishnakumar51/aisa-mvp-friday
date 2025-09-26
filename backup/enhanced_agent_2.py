# agents/enhanced_agent_2.py

from pathlib import Path
import json
import re
from fastapi import HTTPException
from typing import TypedDict, List, Optional, Dict, Any, Tuple
import importlib.util

from config import ARTIFACTS_DIR
from agents.llm_utils import get_llm_response, get_langchain_llm

from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Enhanced Pydantic models
class BlueprintStep(BaseModel):
    step_id: int
    screen_name: str
    description: str
    action: str
    target_element_description: str
    value_to_enter: Optional[str] = None
    associated_image: Optional[str] = None

class BlueprintSummary(BaseModel):
    overall_goal: str
    target_application: str
    platform: str

class AutomationBlueprint(BaseModel):
    summary: BlueprintSummary
    steps: List[BlueprintStep]

class CodeAnalysis(BaseModel):
    complexity_score: int = Field(description="Complexity rating from 1-10", ge=1, le=10)
    recommended_patterns: List[str] = Field(description="List of recommended design patterns")
    potential_issues: List[str] = Field(description="List of potential issues or risks")
    optimization_suggestions: List[str] = Field(description="Performance and reliability optimizations")

class EnhancedCodeOutput(BaseModel):
    python_code: str = Field(description="Complete, production-ready Python script with advanced patterns")
    requirements: str = Field(description="Comprehensive requirements.txt with pinned versions")
    code_analysis: CodeAnalysis = Field(description="Analysis of the generated code")
    test_code: Optional[str] = Field(description="Optional basic test suite")
    documentation: str = Field(description="Technical documentation and usage guide")

class Agent2Output(TypedDict):
    script: str
    requirements: str
    analysis: dict
    test_script: Optional[str]
    documentation: str

# Advanced setup templates with modern best practices
MOBILE_SETUP_TEMPLATE = """
import os
import sys
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from contextlib import contextmanager
import subprocess
import json
from dataclasses import dataclass
from enum import Enum

from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.options.ios import XCUITestOptions
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, 
    NoSuchElementException, 
    StaleElementReferenceException,
    WebDriverException
)
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.actions import interaction
from faker import Faker

class Platform(Enum):
    ANDROID = "android"
    IOS = "ios"

@dataclass
class DeviceConfig:
    platform: Platform
    device_name: str
    platform_version: str
    app_package: Optional[str] = None
    app_activity: Optional[str] = None
    bundle_id: Optional[str] = None
    udid: Optional[str] = None

class EnhancedMobileDriver:
    \"\"\"Advanced mobile driver with retry mechanisms, logging, and error handling\"\"\"
    
    def __init__(self, device_config: DeviceConfig, server_url: str = "http://127.0.0.1:4723"):
        self.config = device_config
        self.server_url = server_url
        self.driver: Optional[webdriver.Remote] = None
        self.wait: Optional[WebDriverWait] = None
        self.screen_size: Optional[Dict[str, int]] = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_driver(self, timeout: int = 30) -> bool:
        \"\"\"Enhanced driver setup with comprehensive error handling\"\"\"
        try:
            self.logger.info("Setting up mobile driver...")
            
            if self.config.platform == Platform.ANDROID:
                options = UiAutomator2Options()
                options.platform_name = 'Android'
                options.device_name = self.config.device_name
                options.platform_version = self.config.platform_version
                options.automation_name = 'UiAutomator2'
                
                if self.config.app_package:
                    options.app_package = self.config.app_package
                if self.config.app_activity:
                    options.app_activity = self.config.app_activity
                    
                # Enhanced capabilities
                options.no_reset = False
                options.full_reset = False
                options.new_command_timeout = 300
                options.unicode_keyboard = True
                options.reset_keyboard = True
                options.auto_grant_permissions = True
                options.skip_unlock = True
                options.ignore_hidden_api_policy_error = True
                
            elif self.config.platform == Platform.IOS:
                options = XCUITestOptions()
                options.platform_name = 'iOS'
                options.device_name = self.config.device_name
                options.platform_version = self.config.platform_version
                options.automation_name = 'XCUITest'
                
                if self.config.bundle_id:
                    options.bundle_id = self.config.bundle_id
                if self.config.udid:
                    options.udid = self.config.udid
            
            self.driver = webdriver.Remote(self.server_url, options=options)
            self.wait = WebDriverWait(self.driver, timeout)
            self.screen_size = self.driver.get_window_size()
            
            # Enhanced settings for reliability
            if self.config.platform == Platform.ANDROID:
                self.driver.update_settings({
                    "enforceXPath1": True,
                    "elementResponseAttributes": "name,text,className,resourceId",
                    "shouldUseCompactResponses": False
                })
            
            self.logger.info(f"Driver ready - Screen size: {self.screen_size}")
            return True
            
        except Exception as e:
            self.logger.error(f"Driver setup failed: {e}")
            return False

    @contextmanager
    def safe_operation(self, operation_name: str, max_retries: int = 3):
        \"\"\"Context manager for safe operations with retry logic\"\"\"
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Attempting {operation_name} (attempt {attempt + 1}/{max_retries})")
                yield attempt
                break
            except StaleElementReferenceException:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Stale element in {operation_name}, retrying...")
                    time.sleep(1)
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Error in {operation_name}: {e}, retrying...")
                    time.sleep(1)
                else:
                    raise

    def find_element_smart(self, locator: Tuple[str, str], timeout: int = 10, 
                          description: str = "") -> Optional[Any]:
        \"\"\"Smart element finding with multiple strategies\"\"\"
        try:
            with self.safe_operation(f"Finding element: {description}"):
                element = self.wait.until(
                    EC.presence_of_element_located(locator),
                    timeout
                )
                # Verify element is interactive
                if element.is_displayed() and element.is_enabled():
                    return element
                return None
        except TimeoutException:
            self.logger.warning(f"Element not found: {description}")
            return None

    def click_smart(self, locator: Tuple[str, str], description: str = "") -> bool:
        \"\"\"Smart clicking with fallback strategies\"\"\"
        with self.safe_operation(f"Clicking: {description}"):
            element = self.find_element_smart(locator, description=description)
            if not element:
                return False
                
            try:
                # Try regular click first
                element.click()
                time.sleep(0.5)
                return True
            except WebDriverException:
                # Fallback to touch action
                try:
                    touch = PointerInput(interaction.POINTER_TOUCH, "touch")
                    action = ActionBuilder(self.driver)
                    action.add_action(touch.move_to_element(element))
                    action.add_action(touch.press())
                    action.add_action(touch.pause(0.1))
                    action.add_action(touch.release())
                    action.perform()
                    time.sleep(0.5)
                    return True
                except Exception as e:
                    self.logger.error(f"Click failed for {description}: {e}")
                    return False

    def type_smart(self, locator: Tuple[str, str], text: str, 
                   description: str = "", clear_first: bool = True) -> bool:
        \"\"\"Smart text input with enhanced reliability\"\"\"
        with self.safe_operation(f"Typing: {description}"):
            element = self.find_element_smart(locator, description=description)
            if not element:
                return False
                
            try:
                element.click()
                time.sleep(0.3)
                
                if clear_first:
                    if self.config.platform == Platform.ANDROID:
                        # Android: Use select all + delete
                        self.driver.press_keycode(29)  # A
                        self.driver.press_keycode(57, 4096)  # CTRL+A  
                        time.sleep(0.2)
                        self.driver.press_keycode(67)  # DEL
                    else:
                        # iOS: Use clear
                        element.clear()
                    time.sleep(0.3)
                
                element.send_keys(text)
                time.sleep(0.5)
                return True
                
            except Exception as e:
                self.logger.error(f"Typing failed for {description}: {e}")
                # ADB fallback for Android
                if self.config.platform == Platform.ANDROID:
                    try:
                        subprocess.run(['adb', 'shell', 'input', 'text', text], 
                                     timeout=10, check=False)
                        return True
                    except:
                        pass
                return False

    def cleanup(self):
        \"\"\"Clean driver shutdown\"\"\"
        if self.driver:
            try:
                self.driver.quit()
                self.logger.info("Driver cleaned up successfully")
            except:
                self.logger.warning("Driver cleanup had issues")

def create_mobile_driver(app_package: str, app_activity: str = None, 
                        device_name: str = "Android", platform_version: str = "11") -> EnhancedMobileDriver:
    \"\"\"Factory function for mobile driver creation\"\"\"
    config = DeviceConfig(
        platform=Platform.ANDROID,
        device_name=device_name,
        platform_version=platform_version,
        app_package=app_package,
        app_activity=app_activity or f"{app_package}.MainActivity"
    )
    return EnhancedMobileDriver(config)
"""

WEB_SETUP_TEMPLATE = """
import os
import sys
import time
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from contextlib import contextmanager
import json
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, Playwright, Page, Browser, BrowserContext, expect
from faker import Faker

class BrowserType(Enum):
    CHROMIUM = "chromium"
    FIREFOX = "firefox" 
    WEBKIT = "webkit"

@dataclass
class WebConfig:
    browser_type: BrowserType = BrowserType.CHROMIUM
    headless: bool = False
    viewport_width: int = 1920
    viewport_height: int = 1080
    user_agent: Optional[str] = None
    timeout: int = 30000
    slow_mo: int = 0

class EnhancedWebDriver:
    \"\"\"Advanced web driver with modern best practices\"\"\"
    
    def __init__(self, config: WebConfig):
        self.config = config
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_browser(self) -> bool:
        \"\"\"Enhanced browser setup with comprehensive configuration\"\"\"
        try:
            self.logger.info("Setting up web browser...")
            
            self.playwright = sync_playwright().start()
            
            # Browser selection with enhanced options
            browser_options = {
                'headless': self.config.headless,
                'slow_mo': self.config.slow_mo,
                'args': [
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-gpu',
                    '--window-size=1920,1080'
                ]
            }
            
            if self.config.browser_type == BrowserType.CHROMIUM:
                self.browser = self.playwright.chromium.launch(**browser_options)
            elif self.config.browser_type == BrowserType.FIREFOX:
                self.browser = self.playwright.firefox.launch(**browser_options)
            elif self.config.browser_type == BrowserType.WEBKIT:
                self.browser = self.playwright.webkit.launch(**browser_options)
            
            # Enhanced context with stealth settings
            context_options = {
                'viewport': {
                    'width': self.config.viewport_width, 
                    'height': self.config.viewport_height
                },
                'user_agent': self.config.user_agent or 
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'java_script_enabled': True,
                'accept_downloads': True,
                'ignore_https_errors': True,
                'extra_http_headers': {
                    'Accept-Language': 'en-US,en;q=0.9'
                }
            }
            
            self.context = self.browser.new_context(**context_options)
            
            # Add stealth scripts
            self.context.add_init_script(\"\"\"
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                
                window.chrome = {
                    runtime: {},
                };
                
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });
            \"\"\")
            
            self.page = self.context.new_page()
            self.page.set_default_timeout(self.config.timeout)
            
            self.logger.info("Browser setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Browser setup failed: {e}")
            return False

    @contextmanager
    def safe_operation(self, operation_name: str, max_retries: int = 3):
        \"\"\"Context manager for safe operations\"\"\"
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Attempting {operation_name} (attempt {attempt + 1}/{max_retries})")
                yield attempt
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Error in {operation_name}: {e}, retrying...")
                    time.sleep(1)
                else:
                    raise

    def navigate_smart(self, url: str, wait_until: str = "networkidle") -> bool:
        \"\"\"Smart navigation with enhanced error handling\"\"\"
        try:
            with self.safe_operation("Navigation"):
                # Validate URL
                parsed = urlparse(url)
                if not parsed.scheme:
                    url = f"https://{url}"
                
                self.page.goto(url, wait_until=wait_until, timeout=self.config.timeout)
                self.logger.info(f"Successfully navigated to: {url}")
                return True
                
        except Exception as e:
            self.logger.error(f"Navigation failed: {e}")
            return False

    def find_element_smart(self, selector: str, timeout: int = None) -> Optional[Any]:
        \"\"\"Smart element finding with multiple selector strategies\"\"\"
        timeout = timeout or self.config.timeout
        
        # Try different selector strategies
        selectors = [
            selector,  # Original selector
            f"[data-testid*='{selector}']",  # Test ID
            f"[aria-label*='{selector}']",  # Accessibility
            f"text='{selector}'",  # Text content
            f":has-text('{selector}')"  # Contains text
        ]
        
        for sel in selectors:
            try:
                element = self.page.locator(sel).first
                if element.is_visible(timeout=timeout):
                    return element
            except:
                continue
        
        return None

    def click_smart(self, selector: str, description: str = "") -> bool:
        \"\"\"Smart clicking with fallback strategies\"\"\"
        try:
            with self.safe_operation(f"Clicking: {description}"):
                element = self.find_element_smart(selector)
                if not element:
                    return False
                
                # Try different click methods
                try:
                    element.click(timeout=5000)
                    return True
                except:
                    # Force click
                    element.click(force=True, timeout=5000)
                    return True
                    
        except Exception as e:
            self.logger.error(f"Click failed for {description}: {e}")
            return False

    def type_smart(self, selector: str, text: str, description: str = "", 
                   clear_first: bool = True) -> bool:
        \"\"\"Smart text input\"\"\"
        try:
            with self.safe_operation(f"Typing: {description}"):
                element = self.find_element_smart(selector)
                if not element:
                    return False
                
                if clear_first:
                    element.clear()
                    time.sleep(0.2)
                
                element.fill(text)
                time.sleep(0.3)
                return True
                
        except Exception as e:
            self.logger.error(f"Typing failed for {description}: {e}")
            return False

    def wait_for_element(self, selector: str, state: str = "visible", 
                        timeout: int = None) -> bool:
        \"\"\"Wait for element with different states\"\"\"
        timeout = timeout or self.config.timeout
        try:
            self.page.locator(selector).wait_for(state=state, timeout=timeout)
            return True
        except:
            return False

    def cleanup(self):
        \"\"\"Clean browser shutdown\"\"\"
        try:
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            self.logger.info("Browser cleaned up successfully")
        except:
            self.logger.warning("Browser cleanup had issues")

def create_web_driver(headless: bool = False, browser_type: str = "chromium") -> EnhancedWebDriver:
    \"\"\"Factory function for web driver creation\"\"\"
    config = WebConfig(
        browser_type=BrowserType(browser_type.lower()),
        headless=headless
    )
    return EnhancedWebDriver(config)
"""

def analyze_blueprint_complexity(blueprint: AutomationBlueprint) -> CodeAnalysis:
    """Analyze blueprint to determine code complexity and patterns"""
    
    complexity_factors = 0
    patterns = set()
    issues = []
    optimizations = []
    
    # Analyze steps for complexity
    for step in blueprint.steps:
        # Action complexity
        if step.action in ['press_and_hold', 'scroll', 'swipe']:
            complexity_factors += 2
        elif step.action in ['click', 'type_text']:
            complexity_factors += 1
        
        # Element complexity
        if any(word in step.target_element_description.lower() 
               for word in ['dropdown', 'calendar', 'picker', 'dialog']):
            complexity_factors += 2
            patterns.add("complex_ui_elements")
        
        # Dynamic content
        if any(word in step.description.lower() 
               for word in ['wait', 'load', 'appear', 'dynamic']):
            complexity_factors += 1
            patterns.add("dynamic_content_handling")
        
        # CAPTCHA or verification
        if any(word in step.description.lower() 
               for word in ['captcha', 'verification', 'challenge']):
            complexity_factors += 3
            patterns.add("verification_handling")
            issues.append("CAPTCHA handling may require manual intervention")
    
    # Platform-specific patterns
    if blueprint.summary.platform == "mobile":
        patterns.add("mobile_gestures")
        patterns.add("device_specific_handling")
        optimizations.append("Use native mobile gestures for better reliability")
    else:
        patterns.add("web_interactions")
        patterns.add("cross_browser_compatibility")
        optimizations.append("Implement responsive design handling")
    
    # General patterns
    patterns.add("page_object_model")
    patterns.add("explicit_waits")
    patterns.add("retry_mechanisms")
    
    # Determine final complexity score
    complexity_score = min(max(complexity_factors // 2 + 2, 1), 10)
    
    # Add optimizations based on complexity
    if complexity_score >= 7:
        optimizations.extend([
            "Implement comprehensive error handling",
            "Add performance monitoring",
            "Use parallel execution where possible"
        ])
    
    optimizations.extend([
        "Add detailed logging for debugging",
        "Implement element visibility checks",
        "Use configuration files for test data"
    ])
    
    return CodeAnalysis(
        complexity_score=complexity_score,
        recommended_patterns=list(patterns),
        potential_issues=issues,
        optimization_suggestions=optimizations
    )

def generate_advanced_requirements(platform: str, blueprint: AutomationBlueprint, 
                                 analysis: CodeAnalysis) -> str:
    """Generate comprehensive requirements.txt with pinned versions"""
    
    base_requirements = {
        "mobile": [
            "appium-python-client==2.12.1",
            "selenium==4.16.0",
            "Appium-Python-Client==2.12.1"
        ],
        "web": [
            "playwright==1.40.0",
            "selenium==4.16.0"
        ]
    }
    
    # Common requirements
    common_reqs = [
        "faker==22.0.0",
        "pytest==7.4.3",
        "pytest-html==4.1.1", 
        "pytest-xdist==3.5.0",
        "allure-pytest==2.13.2",
        "pydantic==2.5.0",
        "requests==2.31.0",
        "python-dotenv==1.0.0",
        "retrying==1.3.4",
        "pillow==10.1.0"
    ]
    
    # Analysis-based requirements
    analysis_reqs = []
    if "verification_handling" in analysis.recommended_patterns:
        analysis_reqs.extend([
            "opencv-python==4.8.1.78",
            "pytesseract==0.3.10"
        ])
    
    if analysis.complexity_score >= 7:
        analysis_reqs.extend([
            "structlog==23.2.0",
            "datadog==0.48.0"
        ])
    
    # Platform-specific additions
    platform_reqs = base_requirements.get(platform, [])
    
    # Combine all requirements
    all_reqs = platform_reqs + common_reqs + analysis_reqs
    
    # Sort and deduplicate
    unique_reqs = sorted(list(set(all_reqs)))
    
    return "\n".join(unique_reqs)

def create_advanced_system_prompt(platform: str, analysis: CodeAnalysis) -> str:
    """Create advanced system prompt based on complexity analysis"""
    
    base_prompt = f"""You are an elite automation engineer and code architect specializing in {platform} automation.
Your expertise encompasses:

🔧 TECHNICAL EXCELLENCE:
- Modern automation frameworks (Playwright, Appium, Selenium WebDriver)
- Advanced design patterns (Page Object Model, Factory Pattern, Builder Pattern)
- Production-grade error handling and retry mechanisms
- Performance optimization and parallel execution
- CI/CD integration and test reporting

🏗️ ARCHITECTURE PRINCIPLES:
- Clean Code principles and SOLID design
- Dependency Injection and Inversion of Control  
- Async/await patterns for performance
- Comprehensive logging and monitoring
- Configuration management and environment handling

🧪 TESTING BEST PRACTICES:
- Test data management and faker integration
- Cross-browser/device compatibility
- Accessibility testing considerations
- Visual regression testing
- API integration testing

📊 CODE ANALYSIS INSIGHTS:
Complexity Score: {analysis.complexity_score}/10
Recommended Patterns: {', '.join(analysis.recommended_patterns)}
Key Optimizations: {', '.join(analysis.optimization_suggestions[:3])}

🎯 GENERATION REQUIREMENTS:
1. Write production-ready, maintainable code with proper error handling
2. Implement the mandatory setup code exactly as provided
3. Use advanced locator strategies with fallbacks
4. Add comprehensive logging and debugging capabilities
5. Include retry mechanisms for flaky elements
6. Follow modern Python best practices (type hints, dataclasses, enums)
7. Implement smart waits and element validation
8. Add performance monitoring and reporting
9. Use faker library for realistic test data generation
10. Create modular, reusable code components

Generate code that a senior automation engineer would be proud to deploy to production."""

    if analysis.complexity_score >= 7:
        base_prompt += f"""

⚠️ HIGH COMPLEXITY DETECTED:
This automation involves complex interactions. Implement:
- Advanced retry logic with exponential backoff
- Multiple element finding strategies  
- Comprehensive error recovery mechanisms
- Detailed step-by-step logging
- Performance benchmarking
- Fallback interaction methods
"""

    return base_prompt

def run_enhanced_agent2(seq_no: str, blueprint_dict: dict) -> Agent2Output:
    """Enhanced Agent 2 with advanced code generation capabilities"""
    print(f"[{seq_no}] 🚀 Running Enhanced Agent 2: Advanced Code Generation")
    
    try:
        blueprint = AutomationBlueprint(**blueprint_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid blueprint structure: {e}")

    # Create output directory
    out_dir = ARTIFACTS_DIR / seq_no / "enhanced_agent2"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze blueprint complexity
    print(f"[{seq_no}] 🔍 Analyzing blueprint complexity...")
    analysis = analyze_blueprint_complexity(blueprint)
    print(f"[{seq_no}] 📊 Complexity Score: {analysis.complexity_score}/10")
    print(f"[{seq_no}] 🏗️ Recommended Patterns: {', '.join(analysis.recommended_patterns[:3])}")
    
    # Determine platform and setup
    platform = blueprint.summary.platform.lower()
    framework = "Enhanced Appium with Selenium" if platform == "mobile" else "Enhanced Playwright"
    setup_code = MOBILE_SETUP_TEMPLATE if platform == "mobile" else WEB_SETUP_TEMPLATE
    
    print(f"[{seq_no}] 🛠️ Using {framework} framework")

    # Collect image descriptions with enhanced analysis
    image_descriptions: Dict[str, str] = {}
    for step in blueprint.steps:
        if step.associated_image:
            img_path = ARTIFACTS_DIR / seq_no / "agent1" / step.associated_image
            if img_path.exists():
                enhanced_prompt = f"""Analyze this screenshot for automation scripting. Focus on:
1. Exact element descriptions (buttons, inputs, labels)
2. UI patterns and layout structure
3. Potential accessibility attributes
4. Dynamic content indicators
5. Loading states or animations
6. Error states or validation messages
Provide detailed technical description for robust element identification."""
                
                description = get_llm_response(
                    enhanced_prompt,
                    "You are an expert UI/UX analyst specializing in automation.",
                    images=[img_path]
                )
                image_descriptions[img_path.name] = description

    # Get enhanced LLM
    llm = get_langchain_llm()
    
    # Create structured output with proper tool_choice format
    structured_llm = llm.with_structured_output(EnhancedCodeOutput)

    # Create advanced system prompt
    system_prompt = create_advanced_system_prompt(platform, analysis)
    
    # Enhanced human prompt template
    human_prompt_template = """
🎯 AUTOMATION GENERATION REQUEST

**Blueprint Analysis:**
Platform: {platform}
Complexity Score: {complexity_score}/10
Target Application: {target_app}
Total Steps: {step_count}

**Automation Blueprint:**
```json
{blueprint_json}
```

**Visual Context (Screenshots Analysis):**
```json
{image_descriptions_json}
```

**Code Analysis Insights:**
- Recommended Patterns: {patterns}
- Potential Issues: {issues}  
- Optimizations: {optimizations}

**MANDATORY Setup Code (Must be included exactly):**
```python
{setup_code}
```

**GENERATION REQUIREMENTS:**
1. Create a complete, production-ready automation script
2. Include the mandatory setup code exactly as provided
3. Implement advanced error handling and retry mechanisms
4. Use smart element finding with multiple fallback strategies
5. Add comprehensive logging and debugging capabilities
6. Generate realistic test data using faker
7. Include performance monitoring and timing
8. Create modular, maintainable code structure
9. Add proper type hints and documentation
10. Implement graceful cleanup and resource management

Generate production-quality code that handles edge cases and provides excellent debugging information.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt_template),
    ])
    
    chain = prompt | structured_llm
    
    try:
        print(f"[{seq_no}] Invoking enhanced structured output chain...")
        
        input_variables = {
            "platform": platform,
            "complexity_score": analysis.complexity_score,
            "target_app": blueprint.summary.target_application,
            "step_count": len(blueprint.steps),
            "blueprint_json": blueprint.model_dump_json(indent=2),
            "image_descriptions_json": json.dumps(image_descriptions, indent=2),
            "patterns": ", ".join(analysis.recommended_patterns),
            "issues": ", ".join(analysis.potential_issues) or "None identified",
            "optimizations": ", ".join(analysis.optimization_suggestions),
            "setup_code": setup_code
        }

        print(f"[{seq_no}] Generating advanced code with complexity score {analysis.complexity_score}...")
        code_output: EnhancedCodeOutput = chain.invoke(input_variables)
        
        # Validate output
        if not code_output.python_code or not code_output.requirements:
            raise ValueError("LLM response missing essential code or requirements content")

        # Generate enhanced requirements
        enhanced_requirements = generate_advanced_requirements(platform, blueprint, analysis)
        
        # Combine generated requirements with enhanced ones
        final_requirements = f"# Generated requirements\n{code_output.requirements}\n\n# Enhanced requirements\n{enhanced_requirements}"
        
        # Save all outputs
        script_path = out_dir / "automation_script.py"
        reqs_path = out_dir / "requirements.txt"
        analysis_path = out_dir / "code_analysis.json"
        docs_path = out_dir / "documentation.md"
        
        script_path.write_text(code_output.python_code, encoding="utf-8")
        reqs_path.write_text(final_requirements, encoding="utf-8")
        
        # Save analysis
        analysis_data = {
            "complexity_score": code_output.code_analysis.complexity_score,
            "recommended_patterns": code_output.code_analysis.recommended_patterns,
            "potential_issues": code_output.code_analysis.potential_issues,
            "optimization_suggestions": code_output.code_analysis.optimization_suggestions,
            "blueprint_summary": blueprint.summary.model_dump(),
            "generation_metadata": {
                "platform": platform,
                "framework": framework,
                "steps_count": len(blueprint.steps),
                "images_processed": len(image_descriptions)
            }
        }
        analysis_path.write_text(json.dumps(analysis_data, indent=2), encoding="utf-8")
        
        # Save documentation
        docs_path.write_text(code_output.documentation, encoding="utf-8")
        
        # Save test code if generated
        test_script_path = None
        if code_output.test_code:
            test_script_path = out_dir / "test_automation.py"
            test_script_path.write_text(code_output.test_code, encoding="utf-8")
        
        print(f"[{seq_no}] Enhanced Agent 2 completed successfully!")
        print(f"[{seq_no}] Generated files:")
        print(f"  - Automation Script: {script_path}")
        print(f"  - Requirements: {reqs_path}")
        print(f"  - Analysis: {analysis_path}")
        print(f"  - Documentation: {docs_path}")
        if test_script_path:
            print(f"  - Test Suite: {test_script_path}")
        
        return {
            "script": str(script_path),
            "requirements": str(reqs_path),
            "analysis": analysis_data,
            "test_script": str(test_script_path) if test_script_path else None,
            "documentation": str(docs_path)
        }
        
    except Exception as e:
        print(f"[{seq_no}] Enhanced Agent 2 failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced Agent 2 failed: {e}")

# Additional helper functions for web search integration
def search_automation_patterns(query: str) -> str:
    """Search for latest automation patterns and best practices"""
    try:
        # This would integrate with web search to get latest patterns
        # For now, return cached best practices
        patterns_db = {
            "mobile_automation": """
            Latest mobile automation best practices:
            - Use native gestures over coordinate-based actions
            - Implement smart waits with multiple conditions
            - Handle different screen densities and resolutions
            - Use accessibility IDs when available
            - Implement retry mechanisms for network-dependent actions
            """,
            "web_automation": """
            Latest web automation best practices:
            - Use Playwright for modern web apps
            - Implement page object model with async/await
            - Handle dynamic content with proper waits
            - Use data-testid attributes for reliable selectors
            - Implement visual regression testing
            """,
            "error_handling": """
            Advanced error handling patterns:
            - Implement exponential backoff for retries
            - Use circuit breaker pattern for external dependencies
            - Add comprehensive logging with structured data
            - Implement graceful degradation for optional steps
            - Use health checks and monitoring
            """
        }
        
        return patterns_db.get(query, "No specific patterns found")
    except:
        return "Search unavailable"

def validate_generated_code(code: str, platform: str) -> List[str]:
    """Validate generated code for common issues"""
    issues = []
    
    # Check for common anti-patterns
    if "time.sleep(" in code and code.count("time.sleep(") > 5:
        issues.append("Excessive use of time.sleep() - consider using explicit waits")
    
    if "find_element(" in code and "WebDriverWait" not in code:
        issues.append("Using find_element without explicit waits")
    
    if platform == "mobile" and "driver.quit()" not in code:
        issues.append("Missing driver cleanup in mobile automation")
    
    if platform == "web" and ".close()" not in code:
        issues.append("Missing browser cleanup in web automation")
    
    # Check for logging
    if "logging" not in code and "print(" in code:
        issues.append("Consider using proper logging instead of print statements")
    
    return issues

# Export the main function
__all__ = ["run_enhanced_agent2", "analyze_blueprint_complexity", "generate_advanced_requirements"]