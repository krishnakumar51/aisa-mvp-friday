import playwright.sync_api as pw
import time
import os
import argparse
import pyautogui
import threading



def create_outlook_account(playwright: pw.Playwright, username: str, password: str, birth_month: str, birth_day: str, birth_year: str, first_name: str, last_name: str) -> None:
    """
    Navigates to the Outlook account creation page, fills in the email username, password, birth date, first name, last name, and clicks next
    
    Args:
        playwright (pw.Playwright): The Playwright instance
        username (str): The desired username part of the email (e.g., 'pranav910' for 'pranav910@outlook.com')
        password (str): The password for the new account
        birth_month (str): The birth month (e.g., 'January', 'February', etc.)
        birth_day (str): The birth day (e.g., '1', '15', '31')
        birth_year (str): The birth year (e.g., '1990', '1995')
        first_name (str): The first name for the account
        last_name (str): The last name for the account
    """
    # Launch browser with additional settings
    browser = playwright.chromium.launch(
        headless=False,
        args=[
            '--no-first-run',
            '--no-service-autorun', 
            '--no-default-browser-check',
            '--disable-dev-shm-usage',
            '--disable-gpu',
            '--no-sandbox',
            '--disable-web-security',
            '--disable-features=VizDisplayCompositor',
            '--disable-blink-features=AutomationControlled',
            '--disable-extensions',
            '--disable-plugins',
            '--disable-default-apps',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding',
            '--disable-features=TranslateUI',
            '--disable-ipc-flooding-protection',
            '--disable-hang-monitor',
            '--disable-prompt-on-repost',
            '--disable-sync',
            '--disable-translate',
            '--metrics-recording-only',
            '--no-report-upload',
            '--safebrowsing-disable-auto-update',
            '--enable-automation=false',
            '--disable-client-side-phishing-detection'
        ]
    )
    
    # Create context with specific viewport and user agent
    context = browser.new_context(
        # viewport={'width': 1366, 'height': 768},
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        locale='en-US',
        timezone_id='America/New_York',
        permissions=['geolocation', 'notifications'],
        extra_http_headers={
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-User': '?1',
            'Sec-Fetch-Dest': 'document'
        }
    )

    # Open new page with additional timeout
    page = context.new_page()
    # stealth_sync(page)  # Apply stealth techniques to avoid bot detection
    page.set_default_timeout(60000)  # Increase default timeout to 60 seconds
    


    page.add_init_script("""
        // Remove webdriver property completely
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
            set: () => {},
            configurable: true,
            enumerable: false
        });
        
        // Override the chrome property with realistic values
        window.chrome = {
            runtime: {
                onConnect: undefined,
                onMessage: undefined
            },
            loadTimes: function() {
                return {
                    requestTime: Date.now() / 1000 - Math.random() * 1000,
                    startLoadTime: Date.now() / 1000 - Math.random() * 1000,
                    commitLoadTime: Date.now() / 1000 - Math.random() * 1000,
                    finishDocumentLoadTime: Date.now() / 1000 - Math.random() * 1000,
                    finishLoadTime: Date.now() / 1000 - Math.random() * 1000,
                    firstPaintTime: Date.now() / 1000 - Math.random() * 1000,
                    firstPaintAfterLoadTime: 0,
                    navigationType: 'Other',
                    wasFetchedViaSpdy: false,
                    wasNpnNegotiated: false,
                    npnNegotiatedProtocol: 'unknown',
                    wasAlternateProtocolAvailable: false,
                    connectionInfo: 'unknown'
                };
            }
        };
        
        // Mock plugins array with realistic plugins
        Object.defineProperty(navigator, 'plugins', {
            get: () => [
                {
                    0: {type: "application/x-google-chrome-pdf", suffixes: "pdf", description: "Portable Document Format", __proto__: MimeType.prototype},
                    description: "Portable Document Format",
                    filename: "internal-pdf-viewer",
                    length: 1,
                    name: "Chrome PDF Plugin"
                },
                {
                    0: {type: "application/pdf", suffixes: "pdf", description: "", __proto__: MimeType.prototype},
                    description: "",
                    filename: "mhjfbmdgcfjbbpaeojofohoefgiehjai",
                    length: 1,
                    name: "Chrome PDF Viewer"
                }
            ],
            configurable: true,
            enumerable: false
        });
        
        // Mock languages
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
            configurable: true,
            enumerable: false
        });
        
        // Override permissions query
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
        );
        
        // Add realistic connection info
        Object.defineProperty(navigator, 'connection', {
            get: () => ({
                effectiveType: '4g',
                rtt: 50,
                downlink: 10,
                saveData: false
            }),
            configurable: true,
            enumerable: false
        });
        
        // Override the prototype for detection evasion
        Object.setPrototypeOf(navigator, Navigator.prototype);
        
        // Remove automation-related properties
        delete navigator.__proto__.webdriver;
        
        // Mock hardwareConcurrency
        Object.defineProperty(navigator, 'hardwareConcurrency', {
            get: () => 4,
            configurable: true,
            enumerable: false
        });
        
        // Mock deviceMemory
        Object.defineProperty(navigator, 'deviceMemory', {
            get: () => 8,
            configurable: true,
            enumerable: false
        });
        
        // Override toString methods to hide automation
        Function.prototype.toString = new Proxy(Function.prototype.toString, {
            apply: function(target, thisArg, argumentsList) {
                if (thisArg && thisArg.name && thisArg.name.includes('automation')) {
                    return 'function () { [native code] }';
                }
                return target.apply(thisArg, argumentsList);
            }
        });
    """)
    # Screenshot functionality removed per user request
    
    try:
        # Navigate directly to Microsoft account signup page
        print("Navigating directly to Microsoft account signup page...")
        page.goto("https://signup.live.com/signup?wa=wsignin1.0&rpsnv=13&ct=1632423730&rver=7.0&wp=MBI_SSL&wreply=https%3A%2F%2Foutlook.live.com%2Fowa%2F&id=292841&CBCXT=out&lw=1&fl=dob%2Cflname%2Cwld", wait_until="domcontentloaded")
        
        # Wait for page to load
        page.wait_for_timeout(5000)


        # Test shadow DOM prevention first
        print("🔍 Testing shadow DOM prevention...")
        shadow_prevented = page.evaluate("""
            () => {
                const captcha = document.querySelector('#px-captcha');
                if (!captcha) {
                    return {status: 'NO_CAPTCHA_FOUND', message: 'Captcha element not present - good!'};
                }
                
                const hasShadowRoot = captcha.shadowRoot !== null;
                return {
                    status: hasShadowRoot ? 'SHADOW_ROOT_EXISTS' : 'SHADOW_ROOT_PREVENTED',
                    hasShadowRoot: hasShadowRoot,
                    message: hasShadowRoot ? 'Shadow DOM detected - automation may fail' : 'Shadow DOM prevented - automation should work'
                };
            }
        """)
        
        print(f"📊 Shadow DOM Test Result: {shadow_prevented['status']}")
        print(f"💬 Message: {shadow_prevented['message']}")
        
        if shadow_prevented.get('hasShadowRoot', False):
            print("❌ WARNING: Shadow DOM detected. Captcha elements may be inaccessible.")
            print("🔧 Consider using different browser arguments or stealth techniques.")
        else:
            print("✅ SUCCESS: Shadow DOM prevented. Captcha elements should be accessible.")

        
        # Debug: Print all input elements found on the page
        print("=== DEBUG: Finding all input elements on page ===")
        inputs = page.query_selector_all("input")
        for i, input_elem in enumerate(inputs):
            try:
                input_type = input_elem.get_attribute("type") or "unknown"
                input_name = input_elem.get_attribute("name") or "unknown"
                input_id = input_elem.get_attribute("id") or "unknown" 
                aria_label = input_elem.get_attribute("aria-label") or "unknown"
                placeholder = input_elem.get_attribute("placeholder") or "unknown"
                visible = input_elem.is_visible()
                print(f"Input {i}: type='{input_type}', name='{input_name}', id='{input_id}', aria-label='{aria_label}', placeholder='{placeholder}', visible={visible}")
            except:
                print(f"Input {i}: Could not get attributes")

        print("=== END DEBUG ===")
        
        # Wait for page to fully stabilize
        # page.wait_for_load_state("networkidle", timeout=30000)
        page.wait_for_load_state("domcontentloaded", timeout=30000)
        page.wait_for_timeout(2000)
        
        # Try a comprehensive approach to find the email input
        print(f"\nAttempting to fill email username: {username}")
        
        # List of comprehensive selectors to try
        email_selectors = [
            # Based on your provided HTML
            "input[aria-label='New email'][type='email']",
            "input[name='New email'][type='email']", 
            "input[id='floatingLabelInput5']",
            
            # Common Microsoft signup selectors
            "input[name='MemberName']",
            "input[name='NewMemberName']",
            "input[name='LiveEmailAddress']",
            
            # Generic email input selectors
            "input[type='email']",
            "input[aria-label*='email' i]",
            "input[placeholder*='email' i]",
            "input[name*='email' i]",
            
            # Any text input that might be the email field
            "input[type='text'][aria-label*='email' i]",
            "input[type='text'][name*='email' i]"
        ]
        
        email_filled = False
        
        # Try each selector
        for selector in email_selectors:
            if email_filled:
                break
                
            try:
                print(f"Trying selector: {selector}")
                email_input = page.wait_for_selector(selector, timeout=5000, state="visible")
                
                if email_input:
                    print(f"Found email input with selector: {selector}")
                    
                    # Scroll into view
                    email_input.scroll_into_view_if_needed()
                    page.wait_for_timeout(1000)
                    
                    # Click to focus
                    email_input.click()
                    page.wait_for_timeout(500)
                    
                    # Clear the field
                    email_input.fill("")
                    page.wait_for_timeout(500)
                    
                    # Type the username
                    email_input.type(username, delay=150)
                    page.wait_for_timeout(1000)
                    
                    # Verify it was filled
                    filled_value = email_input.input_value()
                    if filled_value == username:
                        print(f"Successfully filled email field with: {username}")
                        email_filled = True
                        break
                    else:
                        print(f"Fill verification failed. Expected: {username}, Got: {filled_value}")
                        
            except Exception as e:
                print(f"Selector {selector} failed: {str(e)}")
                continue
                continue
        
        # If standard selectors failed, try JavaScript approach
        if not email_filled:
            print("\nTrying JavaScript approach to find and fill email field...")
            
            filled = page.evaluate(f"""
                () => {{
                    console.log('Starting JavaScript email field search...');
                    
                    // Find all input elements
                    const allInputs = Array.from(document.querySelectorAll('input'));
                    console.log('Found', allInputs.length, 'input elements');
                    
                    // Filter for potential email inputs
                    const emailInputs = allInputs.filter(input => {{
                        const type = (input.type || '').toLowerCase();
                        const name = (input.name || '').toLowerCase();
                        const ariaLabel = (input.getAttribute('aria-label') || '').toLowerCase();
                        const placeholder = (input.placeholder || '').toLowerCase();
                        const id = (input.id || '').toLowerCase();
                        
                        // Check if it's likely an email input
                        const isEmailType = type === 'email';
                        const hasEmailInName = name.includes('email') || name.includes('member');
                        const hasEmailInLabel = ariaLabel.includes('email') || ariaLabel.includes('new email');
                        const hasEmailInPlaceholder = placeholder.includes('email');
                        const hasEmailInId = id.includes('email') || id.includes('floating');
                        
                        const isVisible = input.offsetParent !== null;
                        
                        console.log('Input:', {{
                            type, name, ariaLabel, placeholder, id, 
                            isEmailType, hasEmailInName, hasEmailInLabel, 
                            hasEmailInPlaceholder, hasEmailInId, isVisible
                        }});
                        
                        return isVisible && (isEmailType || hasEmailInName || hasEmailInLabel || 
                                           hasEmailInPlaceholder || hasEmailInId);
                    }});
                    
                    console.log('Found', emailInputs.length, 'potential email inputs');
                    
                    if (emailInputs.length > 0) {{
                        const targetInput = emailInputs[0];
                        console.log('Trying to fill first email input');
                        
                        // Focus and clear
                        targetInput.focus();
                        targetInput.value = '';
                        
                        // Fill with username
                        targetInput.value = '{username}';
                        
                        // Trigger events
                        targetInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        targetInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        targetInput.dispatchEvent(new Event('blur', {{ bubbles: true }}));
                        
                        console.log('Filled with:', targetInput.value);
                        return targetInput.value === '{username}';
                    }}
                    
                    console.log('No suitable email input found');
                    return false;
                }}
            """)
            
            if filled:
                print("Successfully filled email field using JavaScript!")
                email_filled = True
                page.wait_for_timeout(1000)
        
        if not email_filled:
            print("FAILED: Could not find or fill the email input field")
            return
        
        # Now try to find and click the Next button
        print("\nLooking for Next button...")
        
        next_selectors = [
            "input[type='submit']",
            "button[type='submit']", 
            "input[value*='Next' i]",
            "button:has-text('Next')",
            "input[id*='signup' i]",
            "button[id*='signup' i]",
            "#idSIButton9",
            "input[data-report-event='Signup_Submit']",
            "button[data-report-event='Signup_Submit']"
        ]
        
        next_clicked = False
        
        for selector in next_selectors:
            if next_clicked:
                break
                
            try:
                print(f"Trying Next button selector: {selector}")
                next_button = page.wait_for_selector(selector, timeout=5000, state="visible")
                
                if next_button:
                    print(f"Found Next button with selector: {selector}")
                    
                    # Scroll into view and click
                    next_button.scroll_into_view_if_needed()
                    page.wait_for_timeout(1000)
                    next_button.click(force=True)
                    
                    print("Clicked Next button!")
                    next_clicked = True
                    page.wait_for_timeout(3000)
                    break
                    
            except Exception as e:
                print(f"Next button selector {selector} failed: {str(e)}")
                continue
        
        # JavaScript fallback for Next button
        if not next_clicked:
            print("Trying JavaScript approach for Next button...")
            
            clicked = page.evaluate("""
                () => {
                    const buttons = Array.from(document.querySelectorAll('button, input[type="submit"], input[type="button"]'));
                    
                    // Look for buttons with Next-related text or attributes
                    const nextButton = buttons.find(btn => {
                        const text = (btn.innerText || '').toLowerCase();
                        const value = (btn.value || '').toLowerCase();
                        const id = (btn.id || '').toLowerCase();
                        
                        return text.includes('next') || value.includes('next') || 
                               id.includes('signup') || id.includes('next') ||
                               btn.type === 'submit';
                    });
                    
                    if (nextButton) {
                        nextButton.click();
                        return true;
                    }
                    return false;
                }
            """)
            
            if clicked:
                print("Clicked Next button using JavaScript!")
                next_clicked = True
                page.wait_for_timeout(3000)
        
        if not next_clicked:
            print("Could not find or click Next button")
            return
        
        # ----- PASSWORD FILLING SECTION -----
        print(f"\nWaiting for password page to load...")
        page.wait_for_timeout(5000)  # Wait for navigation to password page
        
        # Debug: Print all input elements on the password page
        print("=== DEBUG: Finding password input elements ===")
        password_inputs = page.query_selector_all("input")
        for i, input_elem in enumerate(password_inputs):
            try:
                input_type = input_elem.get_attribute("type") or "unknown"
                input_name = input_elem.get_attribute("name") or "unknown"
                input_id = input_elem.get_attribute("id") or "unknown" 
                aria_label = input_elem.get_attribute("aria-label") or "unknown"
                placeholder = input_elem.get_attribute("placeholder") or "unknown"
                visible = input_elem.is_visible()
                print(f"Password Input {i}: type='{input_type}', name='{input_name}', id='{input_id}', aria-label='{aria_label}', placeholder='{placeholder}', visible={visible}")
            except:
                print(f"Password Input {i}: Could not get attributes")
        print("=== END PASSWORD DEBUG ===")
        
        print(f"Attempting to fill password: {'*' * len(password)}")
        
        # Comprehensive password field selectors
        password_selectors = [
            "input[type='password']",
            "input[name='Password']",
            "input[name='passwd']", 
            "input[name='password']",
            "input[aria-label*='password' i]",
            "input[placeholder*='password' i]",
            "input[id*='password' i]",
            "input[id*='pwd' i]",
            "input[aria-label='Password']",
            "input[name='NewPassword']"
        ]
        
        password_filled = False
        
        # Try each password selector
        for selector in password_selectors:
            if password_filled:
                break
                
            try:
                print(f"Trying password selector: {selector}")
                password_input = page.wait_for_selector(selector, timeout=5000, state="visible")
                
                if password_input:
                    print(f"Found password input with selector: {selector}")
                    
                    # Scroll into view
                    password_input.scroll_into_view_if_needed()
                    page.wait_for_timeout(1000)
                    
                    # Click to focus
                    password_input.click()
                    page.wait_for_timeout(500)
                    
                    # Clear the field
                    password_input.fill("")
                    page.wait_for_timeout(500)
                    
                    # Type the password
                    password_input.type(password, delay=150)
                    page.wait_for_timeout(1000)
                    
                    # Verify it was filled (check length, not actual password for security)
                    filled_value = password_input.input_value()
                    if len(filled_value) == len(password):
                        print(f"Successfully filled password field (length: {len(password)} characters)")
                        password_filled = True
                        break
                    else:
                        print(f"Password fill verification failed. Expected length: {len(password)}, Got length: {len(filled_value)}")
                        
            except Exception as e:
                print(f"Password selector {selector} failed: {str(e)}")
                continue
                continue
        
        # If standard selectors failed, try JavaScript approach for password
        if not password_filled:
            print("\nTrying JavaScript approach to find and fill password field...")
            
            filled = page.evaluate(f"""
                () => {{
                    console.log('Starting JavaScript password field search...');
                    
                    // Find all input elements
                    const allInputs = Array.from(document.querySelectorAll('input'));
                    console.log('Found', allInputs.length, 'input elements');
                    
                    // Filter for password inputs
                    const passwordInputs = allInputs.filter(input => {{
                        const type = (input.type || '').toLowerCase();
                        const name = (input.name || '').toLowerCase();
                        const ariaLabel = (input.getAttribute('aria-label') || '').toLowerCase();
                        const placeholder = (input.placeholder || '').toLowerCase();
                        const id = (input.id || '').toLowerCase();
                        
                        // Check if it's likely a password input
                        const isPasswordType = type === 'password';
                        const hasPasswordInName = name.includes('password') || name.includes('passwd') || name.includes('pwd');
                        const hasPasswordInLabel = ariaLabel.includes('password');
                        const hasPasswordInPlaceholder = placeholder.includes('password');
                        const hasPasswordInId = id.includes('password') || id.includes('pwd');
                        
                        const isVisible = input.offsetParent !== null;
                        
                        console.log('Password Input:', {{
                            type, name, ariaLabel, placeholder, id, 
                            isPasswordType, hasPasswordInName, hasPasswordInLabel, 
                            hasPasswordInPlaceholder, hasPasswordInId, isVisible
                        }});
                        
                        return isVisible && (isPasswordType || hasPasswordInName || hasPasswordInLabel || 
                                           hasPasswordInPlaceholder || hasPasswordInId);
                    }});
                    
                    console.log('Found', passwordInputs.length, 'potential password inputs');
                    
                    if (passwordInputs.length > 0) {{
                        const targetInput = passwordInputs[0];
                        console.log('Trying to fill first password input');
                        
                        // Focus and clear
                        targetInput.focus();
                        targetInput.value = '';
                        
                        // Fill with password
                        targetInput.value = '{password}';
                        
                        // Trigger events
                        targetInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        targetInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        targetInput.dispatchEvent(new Event('blur', {{ bubbles: true }}));
                        
                        console.log('Filled password with length:', targetInput.value.length);
                        return targetInput.value.length === {len(password)};
                    }}
                    
                    console.log('No suitable password input found');
                    return false;
                }}
            """)
            
            if filled:
                print("Successfully filled password field using JavaScript!")
                password_filled = True
                page.wait_for_timeout(1000)
        
        if not password_filled:
            print("FAILED: Could not find or fill the password input field")
            return
        
        # Now try to find and click the Next button on password page
        print("\nLooking for Next button on password page...")
        
        password_next_selectors = [
            "input[type='submit']",
            "button[type='submit']", 
            "input[value*='Next' i]",
            "button:has-text('Next')",
            "input[id*='signup' i]",
            "button[id*='signup' i]",
            "#idSIButton9",
            "input[data-report-event*='Submit' i]",
            "button[data-report-event*='Submit' i]",
            "button:has-text('Create account')",
            "input[value*='Create' i]"
        ]
        
        password_next_clicked = False
        
        for selector in password_next_selectors:
            if password_next_clicked:
                break
                
            try:
                print(f"Trying password Next button selector: {selector}")
                next_button = page.wait_for_selector(selector, timeout=5000, state="visible")
                
                if next_button:
                    print(f"Found password Next button with selector: {selector}")
                    
                    # Scroll into view and click
                    next_button.scroll_into_view_if_needed()
                    page.wait_for_timeout(1000)
                    next_button.click(force=True)
                    
                    print("Clicked password Next button!")
                    password_next_clicked = True
                    page.wait_for_timeout(3000)
                    break
                    
            except Exception as e:
                print(f"Password Next button selector {selector} failed: {str(e)}")
                continue
        
        # JavaScript fallback for password Next button
        if not password_next_clicked:
            print("Trying JavaScript approach for password Next button...")
            
            clicked = page.evaluate("""
                () => {
                    const buttons = Array.from(document.querySelectorAll('button, input[type="submit"], input[type="button"]'));
                    
                    // Look for buttons with Next-related text or submit functionality
                    const nextButton = buttons.find(btn => {
                        const text = (btn.innerText || '').toLowerCase();
                        const value = (btn.value || '').toLowerCase();
                        const id = (btn.id || '').toLowerCase();
                        
                        return text.includes('next') || value.includes('next') || 
                               text.includes('create') || value.includes('create') ||
                               text.includes('continue') || value.includes('continue') ||
                               id.includes('signup') || id.includes('next') ||
                               btn.type === 'submit';
                    });
                    
                    if (nextButton) {
                        nextButton.click();
                        return true;
                    }
                    return false;
                }
            """)
            
            if clicked:
                print("Clicked password Next button using JavaScript!")
                password_next_clicked = True
                page.wait_for_timeout(3000)
        
        if not password_next_clicked:
            print("Could not find or click password Next button")
            return
        
        # ----- BIRTH DATE FILLING SECTION -----
        print(f"\nWaiting for birth date page to load...")
        page.wait_for_timeout(5000)  # Wait for navigation to birth date page
        
        print(f"Attempting to fill birth date: {birth_month} {birth_day}, {birth_year}")
        
        # Debug: Print all buttons and inputs on the birth date page
        print("=== DEBUG: Finding all form elements on birth date page ===")
        all_elements = page.query_selector_all("button, input, select")
        for i, element in enumerate(all_elements):
            try:
                tag_name = element.evaluate("el => el.tagName")
                element_type = element.get_attribute("type") or "unknown"
                element_name = element.get_attribute("name") or "unknown"
                element_id = element.get_attribute("id") or "unknown"
                aria_label = element.get_attribute("aria-label") or "unknown"
                role = element.get_attribute("role") or "unknown"
                visible = element.is_visible()
                inner_text = element.inner_text()[:50] if element.inner_text() else "no text"
                print(f"Element {i}: {tag_name}, type='{element_type}', name='{element_name}', id='{element_id}', aria-label='{aria_label}', role='{role}', visible={visible}, text='{inner_text}'")
            except:
                print(f"Element {i}: Could not get attributes")
        print("=== END BIRTH DATE DEBUG ===")
        
        # Handle Birth Month Dropdown with improved approach
        print("\n--- FILLING BIRTH MONTH ---")
        month_filled = False
        
        # First, try to find the month dropdown button using comprehensive selectors
        month_button = None
        month_selectors = [
            "button[name='BirthMonth']",
            "button[id='BirthMonthDropdown']", 
            "button[aria-label='Birth month']",
            "button[role='combobox'][name='BirthMonth']",
            "button[role='combobox'][aria-label*='month' i]",
            "button[class*='Dropdown'][name='BirthMonth']"
        ]
        
        for selector in month_selectors:
            try:
                print(f"Trying month button selector: {selector}")
                found_button = page.query_selector(selector)
                if found_button and found_button.is_visible():
                    month_button = found_button
                    print(f"Found month button with selector: {selector}")
                    break
            except Exception as e:
                print(f"Month button selector {selector} failed: {str(e)}")
                continue
        
        if month_button:
            try:
                # Click to open the dropdown
                month_button.scroll_into_view_if_needed()
                page.wait_for_timeout(1000)
                print("Clicking month dropdown button...")
                month_button.click(force=True)
                page.wait_for_timeout(2000)  # Give more time for dropdown to open
                
                # Now look for month options with multiple strategies
                print(f"Looking for month option: {birth_month}")
                
                # Strategy 1: Try direct text matching
                month_option = None
                month_option_selectors = [
                    f"*[role='option']:has-text('{birth_month}')",
                    f"div:has-text('{birth_month}')",
                    f"li:has-text('{birth_month}')",
                    f"span:has-text('{birth_month}')",
                    f"button:has-text('{birth_month}')"
                ]
                
                for option_selector in month_option_selectors:
                    try:
                        print(f"Trying month option selector: {option_selector}")
                        found_option = page.wait_for_selector(option_selector, timeout=3000, state="visible")
                        if found_option:
                            month_option = found_option
                            print(f"Found month option with selector: {option_selector}")
                            break
                    except Exception as e:
                        print(f"Month option selector {option_selector} failed: {str(e)}")
                        continue
                
                if month_option:
                    print(f"Clicking on month option: {birth_month}")
                    month_option.click(force=True)
                    month_filled = True
                    page.wait_for_timeout(1000)
                else:
                    # Strategy 2: JavaScript approach with more comprehensive search
                    print("Using JavaScript to find and click month option...")
                    month_selected = page.evaluate(f"""
                        () => {{
                            console.log('Starting JavaScript month selection for: {birth_month}');
                            
                            // Get all visible elements
                            const allElements = Array.from(document.querySelectorAll('*')).filter(el => {{
                                const style = window.getComputedStyle(el);
                                return style.display !== 'none' && style.visibility !== 'hidden' && el.offsetParent !== null;
                            }});
                            
                            console.log('Found', allElements.length, 'visible elements');
                            
                            // Look for elements containing the month name
                            const monthElements = allElements.filter(el => {{
                                const text = el.textContent || el.innerText || '';
                                const exactMatch = text.trim().toLowerCase() === '{birth_month.lower()}';
                                const containsMatch = text.toLowerCase().includes('{birth_month.lower()}') && text.length < 50;
                                return exactMatch || containsMatch;
                            }});
                            
                            console.log('Found', monthElements.length, 'elements containing month text');
                            
                            // Try to click the most suitable element
                            for (const element of monthElements) {{
                                console.log('Trying to click element with text:', element.textContent);
                                try {{
                                    // Check if element is clickable
                                    const isClickable = element.tagName === 'BUTTON' || 
                                                      element.tagName === 'A' || 
                                                      element.tagName === 'DIV' ||
                                                      element.tagName === 'LI' ||
                                                      element.tagName === 'SPAN' ||
                                                      element.onclick ||
                                                      element.getAttribute('role') === 'option' ||
                                                      element.getAttribute('role') === 'button';
                                    
                                    if (isClickable) {{
                                        element.click();
                                        console.log('Successfully clicked month element');
                                        return true;
                                    }}
                                }} catch (e) {{
                                    console.log('Click failed for element:', e);
                                }}
                            }}
                            
                            console.log('No suitable month element found to click');
                            return false;
                        }}
                    """)
                    
                    if month_selected:
                        print(f"Successfully selected month '{birth_month}' using JavaScript")
                        month_filled = True
                        page.wait_for_timeout(1000)
                
            except Exception as e:
                print(f"Error handling month dropdown: {str(e)}")
        
        if not month_filled:
            print(f"FAILED: Could not select birth month '{birth_month}'")
        
        # Handle Birth Day Dropdown with improved approach
        print("\n--- FILLING BIRTH DAY ---")
        day_filled = False
        
        # First, try to find the day dropdown button
        day_button = None
        day_selectors = [
            "button[name='BirthDay']",
            "button[id='BirthDayDropdown']",
            "button[aria-label='Birth day']",
            "button[role='combobox'][name='BirthDay']",
            "button[role='combobox'][aria-label*='day' i]",
            "button[class*='Dropdown'][name='BirthDay']"
        ]
        
        for selector in day_selectors:
            try:
                print(f"Trying day button selector: {selector}")
                found_button = page.query_selector(selector)
                if found_button and found_button.is_visible():
                    day_button = found_button
                    print(f"Found day button with selector: {selector}")
                    break
            except Exception as e:
                print(f"Day button selector {selector} failed: {str(e)}")
                continue
        
        if day_button:
            try:
                # Click to open the dropdown
                day_button.scroll_into_view_if_needed()
                page.wait_for_timeout(1000)
                print("Clicking day dropdown button...")
                day_button.click(force=True)
                page.wait_for_timeout(2000)  # Give more time for dropdown to open
                
                # Now look for day options
                print(f"Looking for day option: {birth_day}")
                
                # Strategy 1: Try direct text matching
                day_option = None
                day_option_selectors = [
                    f"*[role='option']:has-text('{birth_day}')",
                    f"div:has-text('{birth_day}')",
                    f"li:has-text('{birth_day}')",
                    f"span:has-text('{birth_day}')",
                    f"button:has-text('{birth_day}')"
                ]
                
                for option_selector in day_option_selectors:
                    try:
                        print(f"Trying day option selector: {option_selector}")
                        found_option = page.wait_for_selector(option_selector, timeout=3000, state="visible")
                        if found_option:
                            day_option = found_option
                            print(f"Found day option with selector: {option_selector}")
                            break
                    except Exception as e:
                        print(f"Day option selector {option_selector} failed: {str(e)}")
                        continue
                
                if day_option:
                    print(f"Clicking on day option: {birth_day}")
                    day_option.click(force=True)
                    day_filled = True
                    page.wait_for_timeout(1000)
                else:
                    # Strategy 2: JavaScript approach with exact matching
                    print("Using JavaScript to find and click day option...")
                    day_selected = page.evaluate(f"""
                        () => {{
                            console.log('Starting JavaScript day selection for: {birth_day}');
                            
                            // Get all visible elements
                            const allElements = Array.from(document.querySelectorAll('*')).filter(el => {{
                                const style = window.getComputedStyle(el);
                                return style.display !== 'none' && style.visibility !== 'hidden' && el.offsetParent !== null;
                            }});
                            
                            console.log('Found', allElements.length, 'visible elements');
                            
                            // Look for elements with exact day match
                            const dayElements = allElements.filter(el => {{
                                const text = (el.textContent || el.innerText || '').trim();
                                return text === '{birth_day}' || text === '{birth_day.zfill(2)}';
                            }});
                            
                            console.log('Found', dayElements.length, 'elements with day text');
                            
                            // Try to click the most suitable element
                            for (const element of dayElements) {{
                                console.log('Trying to click element with text:', element.textContent);
                                try {{
                                    // Check if element is clickable
                                    const isClickable = element.tagName === 'BUTTON' || 
                                                      element.tagName === 'A' || 
                                                      element.tagName === 'DIV' ||
                                                      element.tagName === 'LI' ||
                                                      element.tagName === 'SPAN' ||
                                                      element.onclick ||
                                                      element.getAttribute('role') === 'option' ||
                                                      element.getAttribute('role') === 'button';
                                    
                                    if (isClickable) {{
                                        element.click();
                                        console.log('Successfully clicked day element');
                                        return true;
                                    }}
                                }} catch (e) {{
                                    console.log('Click failed for element:', e);
                                }}
                            }}
                            
                            console.log('No suitable day element found to click');
                            return false;
                        }}
                    """)
                    
                    if day_selected:
                        print(f"Successfully selected day '{birth_day}' using JavaScript")
                        day_filled = True
                        page.wait_for_timeout(1000)
                
            except Exception as e:
                print(f"Error handling day dropdown: {str(e)}")
        
        if not day_filled:
            print(f"FAILED: Could not select birth day '{birth_day}'")
        
        # Handle Birth Year Input with improved approach
        print("\n--- FILLING BIRTH YEAR ---")
        year_filled = False
        
        # First, try to find the year input field
        year_input = None
        year_selectors = [
            "input[name='BirthYear']",
            "input[id='floatingLabelInput21']", 
            "input[aria-label='Birth year']",
            "input[type='number'][name='BirthYear']",
            "input[type='number'][aria-label*='year' i]",
            "input[placeholder*='year' i]"
        ]
        
        for selector in year_selectors:
            try:
                print(f"Trying year input selector: {selector}")
                found_input = page.query_selector(selector)
                if found_input and found_input.is_visible():
                    year_input = found_input
                    print(f"Found year input with selector: {selector}")
                    break
            except Exception as e:
                print(f"Year input selector {selector} failed: {str(e)}")
                continue
        
        if year_input:
            try:
                # Click to focus and clear
                year_input.scroll_into_view_if_needed()
                page.wait_for_timeout(1000)
                print(f"Filling year input with: {birth_year}")
                
                # Multiple strategies to fill the year
                # Strategy 1: Standard fill
                year_input.click()
                page.wait_for_timeout(500)
                
                # Clear existing content with multiple methods
                year_input.press("Control+a")
                page.wait_for_timeout(200)
                year_input.press("Delete")
                page.wait_for_timeout(500)
                
                # Type the year
                year_input.type(birth_year, delay=150)
                page.wait_for_timeout(1000)
                
                # Verify year was filled
                filled_value = year_input.input_value()
                if filled_value == birth_year:
                    print(f"Successfully filled year: {birth_year}")
                    year_filled = True
                else:
                    print(f"Year verification failed. Expected: {birth_year}, Got: {filled_value}")
                    
                    # Strategy 2: Alternative fill method
                    print("Trying alternative fill method...")
                    year_input.fill("")
                    page.wait_for_timeout(500) 
                    year_input.fill(birth_year)
                    page.wait_for_timeout(1000)
                    
                    filled_value = year_input.input_value()
                    if filled_value == birth_year:
                        print(f"Successfully filled year with alternative method: {birth_year}")
                        year_filled = True
                
            except Exception as e:
                print(f"Error filling year input: {str(e)}")
        
        # JavaScript fallback approach
        if not year_filled:
            print("Trying JavaScript approach for year input...")
            filled = page.evaluate(f"""
                () => {{
                    console.log('Starting JavaScript year filling for: {birth_year}');
                    
                    // Find year input with comprehensive search
                    const allInputs = Array.from(document.querySelectorAll('input'));
                    const yearInputs = allInputs.filter(input => {{
                        const type = (input.type || '').toLowerCase();
                        const name = (input.name || '').toLowerCase();
                        const ariaLabel = (input.getAttribute('aria-label') || '').toLowerCase();
                        const id = (input.id || '').toLowerCase();
                        const placeholder = (input.placeholder || '').toLowerCase();
                        
                        // Check if it looks like a year input
                        const isNumberType = type === 'number';
                        const hasYearInName = name.includes('year');
                        const hasYearInLabel = ariaLabel.includes('year');
                        const hasYearInId = id.includes('year');
                        const hasYearInPlaceholder = placeholder.includes('year');
                        
                        const isVisible = input.offsetParent !== null;
                        
                        console.log('Input check:', {{
                            type, name, ariaLabel, id, placeholder,
                            isNumberType, hasYearInName, hasYearInLabel, hasYearInId, hasYearInPlaceholder, isVisible
                        }});
                        
                        return isVisible && (hasYearInName || hasYearInLabel || hasYearInId || hasYearInPlaceholder || 
                                           (isNumberType && (name.includes('birth') || ariaLabel.includes('birth'))));
                    }});
                    
                    console.log('Found', yearInputs.length, 'potential year inputs');
                    
                    if (yearInputs.length > 0) {{
                        const yearInput = yearInputs[0];
                        console.log('Trying to fill year input');
                        
                        try {{
                            // Focus and clear
                            yearInput.focus();
                            yearInput.select();
                            yearInput.value = '';
                            
                            // Fill with year
                            yearInput.value = '{birth_year}';
                            
                            // Trigger events
                            yearInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            yearInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                            yearInput.dispatchEvent(new Event('blur', {{ bubbles: true }}));
                            
                            console.log('Filled year with value:', yearInput.value);
                            return yearInput.value === '{birth_year}';
                        }} catch (e) {{
                            console.log('Error filling year:', e);
                        }}
                    }}
                    
                    console.log('No suitable year input found');
                    return false;
                }}
            """)
            
            if filled:
                print(f"Successfully filled year '{birth_year}' using JavaScript!")
                year_filled = True
                page.wait_for_timeout(1000)
        
        if not year_filled:
            print(f"FAILED: Could not fill birth year '{birth_year}'")
        
        # Check if all birth date fields were filled successfully
        birth_date_complete = month_filled and day_filled and year_filled
        if not birth_date_complete:
            print("FAILED: Could not complete birth date filling")
            return
        
        print(f"\nBirth date successfully filled: {birth_month} {birth_day}, {birth_year}")
        
        # Try to find and click Next button after birth date
        print("\nLooking for Next button after birth date...")
        
        birthdate_next_selectors = [
            "input[type='submit']",
            "button[type='submit']", 
            "input[value*='Next' i]",
            "button:has-text('Next')",
            "input[id*='signup' i]",
            "button[id*='signup' i]",
            "button:has-text('Create account')",
            "input[value*='Create' i]",
            "button:has-text('Continue')"
        ]
        
        birthdate_next_clicked = False
        
        for selector in birthdate_next_selectors:
            if birthdate_next_clicked:
                break
                
            try:
                print(f"Trying birth date Next button selector: {selector}")
                next_button = page.wait_for_selector(selector, timeout=5000, state="visible")
                
                if next_button:
                    print(f"Found birth date Next button with selector: {selector}")
                    
                    # Scroll into view and click
                    next_button.scroll_into_view_if_needed()
                    page.wait_for_timeout(1000)
                    next_button.click(force=True)
                    
                    print("Clicked birth date Next button!")
                    birthdate_next_clicked = True
                    page.wait_for_timeout(3000)
                    break
                    
            except Exception as e:
                print(f"Birth date Next button selector {selector} failed: {str(e)}")
                continue
        
        # JavaScript fallback for birth date Next button
        if not birthdate_next_clicked:
            print("Trying JavaScript approach for birth date Next button...")
            
            clicked = page.evaluate("""
                () => {
                    const buttons = Array.from(document.querySelectorAll('button, input[type="submit"], input[type="button"]'));
                    
                    const nextButton = buttons.find(btn => {
                        const text = (btn.innerText || '').toLowerCase();
                        const value = (btn.value || '').toLowerCase();
                        const id = (btn.id || '').toLowerCase();
                        
                        return text.includes('next') || value.includes('next') || 
                               text.includes('create') || value.includes('create') ||
                               text.includes('continue') || value.includes('continue') ||
                               id.includes('signup') || id.includes('next') ||
                               btn.type === 'submit';
                    });
                    
                    if (nextButton) {
                        nextButton.click();
                        return true;
                    }
                    return false;
                }
            """)
            
            if clicked:
                print("Clicked birth date Next button using JavaScript!")
                birthdate_next_clicked = True
                page.wait_for_timeout(3000)
        
        if not birthdate_next_clicked:
            print("Could not find or click birth date Next button")
            return
        
        # ----- FIRST NAME AND LAST NAME FILLING SECTION -----
        print(f"\nWaiting for name page to load...")
        page.wait_for_timeout(5000)  # Wait for navigation to name page
        
        print(f"Attempting to fill names: First='{first_name}', Last='{last_name}'")
        
        # Debug: Print all inputs on the name page
        print("=== DEBUG: Finding all input elements on name page ===")
        name_inputs = page.query_selector_all("input")
        for i, input_elem in enumerate(name_inputs):
            try:
                input_type = input_elem.get_attribute("type") or "unknown"
                input_name = input_elem.get_attribute("name") or "unknown"
                input_id = input_elem.get_attribute("id") or "unknown" 
                aria_label = input_elem.get_attribute("aria-label") or "unknown"
                placeholder = input_elem.get_attribute("placeholder") or "unknown"
                visible = input_elem.is_visible()
                print(f"Name Input {i}: type='{input_type}', name='{input_name}', id='{input_id}', aria-label='{aria_label}', placeholder='{placeholder}', visible={visible}")
            except:
                print(f"Name Input {i}: Could not get attributes")
        print("=== END NAME DEBUG ===")
        
        # Handle First Name Input
        print("\n--- FILLING FIRST NAME ---")
        first_name_filled = False
        
        first_name_selectors = [
            "input[id='firstNameInput']",
            "input[name='firstNameInput']",
            "input[id*='firstName' i]",
            "input[name*='firstName' i]",
            "input[aria-label*='first name' i]",
            "input[placeholder*='first name' i]"
        ]
        
        for selector in first_name_selectors:
            if first_name_filled:
                break
                
            try:
                print(f"Trying first name selector: {selector}")
                first_name_input = page.wait_for_selector(selector, timeout=5000, state="visible")
                
                if first_name_input:
                    print(f"Found first name input with selector: {selector}")
                    
                    # Scroll into view and focus
                    first_name_input.scroll_into_view_if_needed()
                    page.wait_for_timeout(1000)
                    first_name_input.click()
                    page.wait_for_timeout(500)
                    
                    # Clear and fill
                    first_name_input.fill("")
                    page.wait_for_timeout(500)
                    first_name_input.type(first_name, delay=150)
                    page.wait_for_timeout(1000)
                    
                    # Verify
                    filled_value = first_name_input.input_value()
                    if filled_value == first_name:
                        print(f"Successfully filled first name: {first_name}")
                        first_name_filled = True
                        break
                    else:
                        print(f"First name verification failed. Expected: {first_name}, Got: {filled_value}")
                        
            except Exception as e:
                print(f"First name selector {selector} failed: {str(e)}")
                continue
        
        # JavaScript fallback for first name
        if not first_name_filled:
            print("Trying JavaScript approach for first name...")
            
            filled = page.evaluate(f"""
                () => {{
                    console.log('Starting JavaScript first name search for: {first_name}');
                    
                    // Find first name input
                    const firstNameInput = document.querySelector('input[id="firstNameInput"]') ||
                                         document.querySelector('input[name="firstNameInput"]') ||
                                         Array.from(document.querySelectorAll('input')).find(input => {{
                                             const id = (input.id || '').toLowerCase();
                                             const name = (input.name || '').toLowerCase();
                                             const ariaLabel = (input.getAttribute('aria-label') || '').toLowerCase();
                                             
                                             return id.includes('firstname') || 
                                                    name.includes('firstname') || 
                                                    ariaLabel.includes('first name') ||
                                                    id.includes('first');
                                         }});
                    
                    if (firstNameInput && firstNameInput.offsetParent !== null) {{
                        console.log('Found first name input, filling...');
                        firstNameInput.focus();
                        firstNameInput.value = '';
                        firstNameInput.value = '{first_name}';
                        
                        // Trigger events
                        firstNameInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        firstNameInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        firstNameInput.dispatchEvent(new Event('blur', {{ bubbles: true }}));
                        
                        console.log('First name filled with:', firstNameInput.value);
                        return firstNameInput.value === '{first_name}';
                    }}
                    
                    console.log('No first name input found');
                    return false;
                }}
            """)
            
            if filled:
                print(f"Successfully filled first name '{first_name}' using JavaScript!")
                first_name_filled = True
                page.wait_for_timeout(1000)
        
        if not first_name_filled:
            print(f"FAILED: Could not fill first name '{first_name}'")
        
        # Handle Last Name Input
        print("\n--- FILLING LAST NAME ---")
        last_name_filled = False
        
        last_name_selectors = [
            "input[id='lastNameInput']",
            "input[name='lastNameInput']", 
            "input[id*='lastName' i]",
            "input[name*='lastName' i]",
            "input[aria-label*='last name' i]",
            "input[placeholder*='last name' i]"
        ]
        
        for selector in last_name_selectors:
            if last_name_filled:
                break
                
            try:
                print(f"Trying last name selector: {selector}")
                last_name_input = page.wait_for_selector(selector, timeout=5000, state="visible")
                
                if last_name_input:
                    print(f"Found last name input with selector: {selector}")
                    
                    # Scroll into view and focus
                    last_name_input.scroll_into_view_if_needed()
                    page.wait_for_timeout(1000)
                    last_name_input.click()
                    page.wait_for_timeout(500)
                    
                    # Clear and fill
                    last_name_input.fill("")
                    page.wait_for_timeout(500)
                    last_name_input.type(last_name, delay=150)
                    page.wait_for_timeout(1000)
                    
                    # Verify
                    filled_value = last_name_input.input_value()
                    if filled_value == last_name:
                        print(f"Successfully filled last name: {last_name}")
                        last_name_filled = True
                        break
                    else:
                        print(f"Last name verification failed. Expected: {last_name}, Got: {filled_value}")
                        
            except Exception as e:
                print(f"Last name selector {selector} failed: {str(e)}")
                continue
        
        # JavaScript fallback for last name
        if not last_name_filled:
            print("Trying JavaScript approach for last name...")
            
            filled = page.evaluate(f"""
                () => {{
                    console.log('Starting JavaScript last name search for: {last_name}');
                    
                    // Find last name input
                    const lastNameInput = document.querySelector('input[id="lastNameInput"]') ||
                                        document.querySelector('input[name="lastNameInput"]') ||
                                        Array.from(document.querySelectorAll('input')).find(input => {{
                                            const id = (input.id || '').toLowerCase();
                                            const name = (input.name || '').toLowerCase();
                                            const ariaLabel = (input.getAttribute('aria-label') || '').toLowerCase();
                                            
                                            return id.includes('lastname') || 
                                                   name.includes('lastname') || 
                                                   ariaLabel.includes('last name') ||
                                                   id.includes('last');
                                        }});
                    
                    if (lastNameInput && lastNameInput.offsetParent !== null) {{
                        console.log('Found last name input, filling...');
                        lastNameInput.focus();
                        lastNameInput.value = '';
                        lastNameInput.value = '{last_name}';
                        
                        // Trigger events
                        lastNameInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        lastNameInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        lastNameInput.dispatchEvent(new Event('blur', {{ bubbles: true }}));
                        
                        console.log('Last name filled with:', lastNameInput.value);
                        return lastNameInput.value === '{last_name}';
                    }}
                    
                    console.log('No last name input found');
                    return false;
                }}
            """)
            
            if filled:
                print(f"Successfully filled last name '{last_name}' using JavaScript!")
                last_name_filled = True
                page.wait_for_timeout(1000)
        
        if not last_name_filled:
            print(f"FAILED: Could not fill last name '{last_name}'")
        
        # Check if both names were filled successfully
        names_complete = first_name_filled and last_name_filled
        if not names_complete:
            print("FAILED: Could not complete name filling")
            return
        
        print(f"\nNames successfully filled: First='{first_name}', Last='{last_name}'")
        
        # Try to find and click Next button after names
        print("\nLooking for Next button after names...")
        
        names_next_selectors = [
            "input[type='submit']",
            "button[type='submit']", 
            "input[value*='Next' i]",
            "button:has-text('Next')",
            "input[id*='signup' i]",
            "button[id*='signup' i]",
            "button:has-text('Create account')",
            "input[value*='Create' i]",
            "button:has-text('Continue')"
        ]
        
        names_next_clicked = False
        
        for selector in names_next_selectors:
            if names_next_clicked:
                break
                
            try:
                print(f"Trying names Next button selector: {selector}")
                next_button = page.wait_for_selector(selector, timeout=5000, state="visible")
                
                if next_button:
                    print(f"Found names Next button with selector: {selector}")
                    
                    # Scroll into view and click
                    next_button.scroll_into_view_if_needed()
                    page.wait_for_timeout(1000)
                    next_button.click(force=True)
                    
                    print("Clicked names Next button!")
                    names_next_clicked = True
                    page.wait_for_timeout(3000)
                    break
                    
            except Exception as e:
                print(f"Names Next button selector {selector} failed: {str(e)}")
                continue
        
        # JavaScript fallback for names Next button
        if not names_next_clicked:
            print("Trying JavaScript approach for names Next button...")
            
            clicked = page.evaluate("""
                () => {
                    const buttons = Array.from(document.querySelectorAll('button, input[type="submit"], input[type="button"]'));
                    
                    const nextButton = buttons.find(btn => {
                        const text = (btn.innerText || '').toLowerCase();
                        const value = (btn.value || '').toLowerCase();
                        const id = (btn.id || '').toLowerCase();
                        
                        return text.includes('next') || value.includes('next') || 
                               text.includes('create') || value.includes('create') ||
                               text.includes('continue') || value.includes('continue') ||
                               id.includes('signup') || id.includes('next') ||
                               btn.type === 'submit';
                    });
                    
                    if (nextButton) {
                        nextButton.click();
                        return true;
                    }
                    return false;
                }
            """)
            
            if clicked:
                print("Clicked names Next button using JavaScript!")
                names_next_clicked = True
                page.wait_for_timeout(3000)
        
        if not names_next_clicked:
            print("Could not find or click names Next button")
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################

        
        
        
        # ----- HUMAN CHALLENGE (PRESS AND HOLD) SECTION -----
        print(f"\nWaiting for human challenge page to load...")
        page.wait_for_timeout(5000)  # Wait for navigation to human challenge page
        current_url = page.url  # Store current URL for comparison
        
        # Look for the iframe-based captcha
        print("=== DEBUG: Looking for iframe-based human captcha ===")
        
        # First, try to find the main iframe containing the captcha
        iframe_selectors = [
            'iframe[data-testid="humanCaptchaIframe"]',
            'iframe[title="Verification challenge"]', 
            'iframe[src*="hsprotect.net"]',
            'iframe[src*="iframe.hsprotect.net"]'
        ]
        
        main_iframe = None
        for selector in iframe_selectors:
            try:
                print(f"Trying iframe selector: {selector}")
                main_iframe = page.wait_for_selector(selector, timeout=10000)
                if main_iframe and main_iframe.is_visible():
                    print(f"✅ Found main captcha iframe with selector: {selector}")
                    break
            except Exception as e:
                print(f"❌ Selector {selector} failed: {e}")
                continue
        
        if not main_iframe:
            print("❌ No iframe captcha found, trying fallback approach...")
            # Fallback: Look for traditional challenge buttons in main page
            challenge_button = None
            captcha_x = None
            captcha_y = None
            
            # Try JavaScript to find any element containing challenge-related text
            challenge_found = page.evaluate("""
                () => {
                    const allElements = Array.from(document.querySelectorAll('div, button, span'));
                    
                    for (const element of allElements) {
                        const text = (element.innerText || '').toLowerCase();
                        const ariaLabel = (element.getAttribute('aria-label') || '').toLowerCase();
                        const id = (element.id || '').toLowerCase();
                        
                        if ((text.includes('press') && text.includes('hold')) ||
                            (ariaLabel.includes('press') && ariaLabel.includes('hold')) ||
                            ariaLabel.includes('human challenge') ||
                            text.includes('human challenge') ||
                            id.includes('challenge')) {
                            
                            if (element.offsetParent !== null) { // is visible
                                element.setAttribute('data-found-challenge', 'true');
                                console.log('Found challenge element:', element);
                                return true;
                            }
                        }
                    }
                    return false;
                }
            """)
            
            if challenge_found:
                challenge_button = page.query_selector("[data-found-challenge='true']")
                print("✅ Found traditional challenge button using JavaScript search!")
                
                # Get coordinates for the challenge button
                if challenge_button:
                    bbox = challenge_button.bounding_box()
                    if bbox:
                        captcha_x = bbox['x'] + bbox['width'] / 2
                        captcha_y = bbox['y'] + bbox['height'] / 2
                        print(f"📍 Challenge button center: ({captcha_x}, {captcha_y})")
                    else:
                        print("❌ Could not get challenge button bounding box")
                        captcha_x = 960  # Default to screen center
                        captcha_y = 540
            
            if not challenge_button or captcha_x is None or captcha_y is None:
                print("❌ FAILED: Could not find any human challenge element or get coordinates.")
                return
                
        else:
            # Found iframe, now handle the iframe-based captcha
            print("✅ Found iframe captcha. Attempting to interact with it...")
            
            # Get iframe bounding box for positioning calculations
            iframe_bbox = main_iframe.bounding_box()
            if not iframe_bbox:
                print("❌ Could not get iframe bounding box")
                return
            
            print(f"📐 Iframe position: x={iframe_bbox['x']}, y={iframe_bbox['y']}, w={iframe_bbox['width']}, h={iframe_bbox['height']}")
            
            # Initialize captcha coordinates with iframe center as default
            captcha_x = iframe_bbox['x'] + iframe_bbox['width'] / 2
            captcha_y = iframe_bbox['y'] + iframe_bbox['height'] / 2
            print(f"📍 Initial captcha coordinates (iframe center): ({captcha_x}, {captcha_y})")
            
            # Try to access iframe content (this may not work due to cross-origin restrictions)
            iframe_content = None
            try:
                iframe_content = main_iframe.content_frame()
                if iframe_content:
                    print("✅ Successfully accessed iframe content!")
                    
                    # Wait for content to load
                    page.wait_for_timeout(3000)
                    
                    # Look for the captcha div inside iframe
                    try:
                        iframe_content.wait_for_selector('#px-captcha', timeout=10000)
                        captcha_element = iframe_content.query_selector('#px-captcha')
                        
                        if captcha_element:
                            print("✅ Found #px-captcha element inside iframe!")
                            
                            # Get captcha element position relative to iframe
                            captcha_bbox = captcha_element.bounding_box()
                            if captcha_bbox:
                                # Calculate absolute position on page
                                captcha_x = iframe_bbox['x'] + captcha_bbox['x'] + captcha_bbox['width'] / 2
                                captcha_y = iframe_bbox['y'] + captcha_bbox['y'] + captcha_bbox['height'] / 2
                                print(f"📍 Refined captcha center: ({captcha_x}, {captcha_y})")
                            else:
                                print(f"⚠️  Could not get captcha element bbox, using iframe center: ({captcha_x}, {captcha_y})")
                        else:
                            print(f"❌ #px-captcha not found, using iframe center: ({captcha_x}, {captcha_y})")
                    
                    except Exception as e:
                        print(f"❌ Error accessing iframe content: {e}")
                        print(f"⚠️  Using iframe center: ({captcha_x}, {captcha_y})")
                else:
                    print("❌ Could not access iframe content (cross-origin restriction)")
                    print(f"⚠️  Using iframe center: ({captcha_x}, {captcha_y})")
                    
            except Exception as e:
                print(f"❌ Error accessing iframe: {e}")
                print(f"⚠️  Using iframe center: ({captcha_x}, {captcha_y})")
        
        # COMMON MOUSE INTERACTION SECTION
        # At this point, we should have valid captcha_x and captcha_y coordinates from either:
        # 1. iframe-based captcha coordinates, or 
        # 2. traditional challenge button coordinates
        
        # Verify that we have valid captcha coordinates
        if captcha_x is None or captcha_y is None:
            print("❌ ERROR: Invalid captcha coordinates!")
            print(f"   captcha_x = {captcha_x}, captcha_y = {captcha_y}")
            return
        
        # Ensure current_url is defined for success detection
        if 'current_url' not in locals():
            current_url = page.url
            
        print(f"✅ Using captcha coordinates: ({captcha_x}, {captcha_y})")
        
        # Get browser window position to convert coordinates
        # We need to account for browser chrome (title bar, address bar, etc.)
        browser_viewport = page.viewport_size
        print(f"Browser viewport size: {browser_viewport}")
        
        # Get current mouse position to help with coordinate conversion
        current_mouse_pos = pyautogui.position()
        print(f"Current physical mouse position: {current_mouse_pos}")
        
        # Use JavaScript to get the browser window position
        browser_info = page.evaluate("""
            () => {
                return {
                    windowX: window.screenX || window.screenLeft || 0,
                    windowY: window.screenY || window.screenTop || 0,
                    innerWidth: window.innerWidth,
                    innerHeight: window.innerHeight,
                    outerWidth: window.outerWidth,
                    outerHeight: window.outerHeight
                };
            }
        """)
        
        print(f"Browser window info: {browser_info}")
        
        # Calculate screen coordinates more accurately
        # Add browser window position + coordinate within page
        screen_x = browser_info['windowX'] + captcha_x
        screen_y = browser_info['windowY'] + captcha_y
        
        print(f"📍 Calculated screen coordinates: ({screen_x}, {screen_y})")
            
        # Safety check: ensure coordinates are within screen bounds
        screen_width, screen_height = pyautogui.size()
        if screen_x < 0 or screen_x > screen_width or screen_y < 0 or screen_y > screen_height:
            print(f"⚠️  Coordinates out of bounds! Screen size: {screen_width}x{screen_height}")
            print(f"⚠️  Using fallback coordinates...")
            # Fallback: use center of screen
            screen_x = screen_width // 2  
            screen_y = screen_height // 2
        
        # Important warning to user
        print("\n" + "="*70)
        print("🚨 IMPORTANT: ABOUT TO TAKE CONTROL OF YOUR PHYSICAL MOUSE!")
        print("="*70)
        print("⚠️  The script will now:")
        print("   1. Move your physical mouse cursor to the captcha")
        print("   2. Press and HOLD the left mouse button physically")
        print("   3. Keep it pressed for exactly 1 minute")
        print("💡 This is REAL hardware control - not simulated!")
        print("💡 The hold will automatically end after 1 minute")
        print("="*70)
        
        # Countdown before taking control
        for i in range(5, 0, -1):
            print(f"⏰ Starting in {i} seconds... (Close browser now to cancel)")
            page.wait_for_timeout(1000)
        
        print(f"\n🖱️  Moving PHYSICAL mouse to captcha position...")
        print(f"📍 Using fixed coordinates: (830, 757)")
        
        # Disable pyautogui failsafe (moving mouse to corner won't stop script)
        pyautogui.FAILSAFE = False
        
        # Move physical mouse to captcha position with smooth movement
        pyautogui.moveTo(830, 757, duration=1.5)
        page.wait_for_timeout(500)
        
        # Press PHYSICAL mouse button down (start holding)
        print("\n🖱️  PHYSICAL MOUSE DOWN - Starting 1-minute press and hold on captcha...")
        print("💡 Using REAL PHYSICAL MOUSE - indistinguishable from human interaction!")
        print("💡 The physical mouse button will stay pressed for exactly 1 minute!")
        print("💡 This is genuine hardware-level mouse control!")
        
        # Press down the physical left mouse button
        try:
            pyautogui.mouseDown(button='left')
            print("✅ Physical left mouse button is now PRESSED and HELD!")
            print("🔒 Mouse button will remain pressed for 1 minute...")
        except Exception as e:
            print(f"❌ Error pressing physical mouse button: {e}")
            print("⚠️  Falling back to Playwright mouse events...")
            # Fallback to Playwright if pyautogui fails
            page.mouse.move(830, 757)
            page.mouse.down(button='left')
            print("⚠️  Using Playwright mouse events as fallback")
        
        # Hold for exactly 1 minute (60 seconds)
        start_time = time.time()
        last_status_time = 0
        hold_duration = 20  # 20 seconds 
        
        print("\n🔄 20-MINUTE HOLD ACTIVE")
            
        try:
            while True:  # Continue for exactly 60 seconds
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check if 1 minute has passed
                if elapsed >= hold_duration:
                    print(f"⏰ 20 minute completed! Releasing mouse button...")
                    break
                
                # Print status every 10 seconds
                if current_time - last_status_time >= 10:
                    remaining = hold_duration - elapsed
                    print(f"⏱️  Still holding... {int(elapsed)}s elapsed, {int(remaining)}s remaining (Mouse button pressed)")
                    last_status_time = current_time
                
                # Check for success indicators every 5 seconds
                if int(elapsed) % 5 == 0:
                    try:
                        # Look for checkmark element (common success indicator)
                        checkmark = page.query_selector("div[id='checkmark']")
                        if checkmark and checkmark.is_visible():
                            print(f"✅ SUCCESS: Checkmark appeared after {int(elapsed)}s!")
                            print("🎉 Challenge completed successfully!")
                            print("💡 Will continue holding until 1 minute is complete...")
                        
                        # Look for other success indicators
                        success_elements = page.query_selector_all("svg[class*='checkmark'], div[class*='success'], div[class*='verified'], div[class*='complete']")
                        for elem in success_elements:
                            if elem.is_visible():
                                print(f"✅ SUCCESS: Success indicator found after {int(elapsed)}s!")
                                print("💡 Will continue holding until 1 minute is complete...")
                                break
                        
                        # Check for page navigation (another success indicator)
                        new_url = page.url
                        if new_url != current_url and "challenge" not in new_url.lower():
                            print(f"✅ SUCCESS: Page navigated after {int(elapsed)}s!")
                            print(f"   New URL: {new_url}")
                            print("💡 Will continue holding until 45 seconds is complete...")
                            current_url = new_url  # Update for future checks
                            
                    except Exception as e:
                        # Continue holding even if there's an error checking
                        # Don't print errors to avoid spam, just continue
                        pass
                
                # Small wait to prevent excessive CPU usage while maintaining the hold
                page.wait_for_timeout(200)  # 200ms intervals
                
        except Exception as e:
            # This will catch browser closure or connection errors
            elapsed = time.time() - start_time
            print(f"\n🛑 Hold ended after {int(elapsed)}s (before 1 minute completion)")
            print(f"   Reason: {str(e)}")
            
            if "Target page, context or browser has been closed" in str(e) or "Connection closed" in str(e):
                print("✅ Browser was closed by user - this is normal!")
            else:
                print("⚠️  Unexpected error occurred")
        
        finally:
            # Always release the physical mouse button when ending
            try:
                print("\n🖱️  Releasing physical mouse button...")
                pyautogui.mouseUp(button='left')
                print("✅ Physical mouse button released successfully!")
            except Exception as e:
                print(f"⚠️  Error releasing mouse button: {e}")


################################################################################################################################                
################################################################################################################################                
################################################################################################################################                
################################################################################################################################                
################################################################################################################################                
################################################################################################################################                
        
        print("\n🏁 Press and hold session completed!")
        print("   The physical mouse button hold has ended.")
        
        # At this point, we have valid captcha_x and captcha_y coordinates from either:
        # 1. The iframe-based captcha (handled above), or  
        # 2. The traditional challenge button (handled in the fallback case)
        # The mouse interaction was handled in the iframe case above.
        # For the traditional button case, we need to add the mouse interaction here.
        
        # Wait for potential navigation after challenge
        page.wait_for_timeout(5000)
        
        # Final wait
        page.wait_for_timeout(5000)
        print("\nProcess completed!")
        print(f"Username '{username}' should create email: {username}@outlook.com")
        print(f"Password has been set (length: {len(password)} characters)")
        print(f"Birth date: {birth_month} {birth_day}, {birth_year}")
        print(f"First name: {first_name}")
        print(f"Last name: {last_name}")
        
        # Return success for backend flow (no interactive blocking)
        return True
        
    except Exception as e:
        print(f"Error during automation: {e}")
        # Propagate error to backend so it can mark account as failed
        raise
    
    finally:
        # Ensure all Playwright resources are cleaned up
        try:
            if 'page' in locals() and page:
                page.close()
        except:
            pass
        try:
            if 'context' in locals() and context:
                context.close()
        except:
            pass
        try:
            if 'browser' in locals() and browser:
                browser.close()
        except:
            pass
