from appium import webdriver
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from appium.options.android import UiAutomator2Options
from selenium.common.exceptions import NoSuchElementException
import os
import pytesseract
from PIL import Image

# Configure Tesseract path for pytesseract. If the hard-coded path doesn't exist,
# try to locate `tesseract` on the system PATH. If not found, print instructions
# and exit with a clear message instead of raising deep exceptions from pytesseract.
pytesseract.pytesseract.tesseract_cmd = r"D:\\Tesseract-OCR\\tesseract.exe"

def _find_tesseract():
    # check configured path first
    configured = pytesseract.pytesseract.tesseract_cmd
    if configured and os.path.exists(configured):
        return configured

    # try to find on PATH
    from shutil import which
    found = which('tesseract')
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
        return found

    return None

tess_path = _find_tesseract()
if not tess_path:
    print('\nTesseract not found. Configure Tesseract OCR before running this script.')
    print('1) Install Tesseract: https://github.com/tesseract-ocr/tesseract')
    print('   On Windows you can use the official installer and note the install path, e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe')
    print('2) Update the variable `pytesseract.pytesseract.tesseract_cmd` in main.py to point to the tesseract executable, or add tesseract to your PATH.')
    print('3) Verify by running: tesseract --version or python -c "import pytesseract; print(pytesseract.pytesseract.tesseract_cmd)"')
    sys.exit(1)

import random
import string
import json
import os
import subprocess
import time
import calendar

from agent import analyze_screen_content

screenshots_folder = "screenshots"

# ensure screenshots folder exists
if not os.path.exists(screenshots_folder):
    os.makedirs(screenshots_folder, exist_ok=True)

for filename in os.listdir(screenshots_folder):
    file_path = os.path.join(screenshots_folder, filename)
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass

try:
    # subprocess.run(["adb", "-s", device_name,"shell", "settings", "put", "global", "http_proxy", "p.webshare.io:9999"])
    device_name = "ZD222GXYPV"
    options = UiAutomator2Options()
    options.platform_name = "Android"
    options.platform_version = "14.0"
    options.udid = device_name
    options.device_name = device_name
    options.app_package = "com.microsoft.office.outlook"
    options.app_activity = "com.microsoft.office.outlook.MainActivity"
    options.automation_name = "UiAutomator2"
    options.auto_grant_permissions = True
    driver = webdriver.Remote("http://127.0.0.1:4723", options=options)
    print("📱 Outlook app launched")
except Exception as e:
    print(f"Error launching Outlook app: {e}")
    exit(1)

def generate_outlook_email():
 
  first = "ivan"
  last = "lopez"

  digits = random.randint(1000000, 9999999)
  email_name = f"{first.lower()}{last.lower()}{digits}"
  return email_name

def generate_password():
    chars = string.ascii_letters + string.digits + "!@#$%"
    password = ''.join(random.choices(chars, k=15))
    return password

def press_button(driver, button_text):
    print(f'Waiting for button with text: {button_text}')
    try:
        button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((AppiumBy.ANDROID_UIAUTOMATOR, f'new UiSelector().text("{button_text}")'))
        )
        button.click()
        return True
    except (NoSuchElementException, Exception) as e:
        print(f"Button '{button_text}' not found: {e}")
        return False
    
def fill_input_field(driver, field_text, input_value):
    try:
        if field_text == 'First name':
            name_fields = driver.find_elements(AppiumBy.CLASS_NAME, "android.widget.EditText")
            name_fields[0].send_keys(input_value)
            return True
        elif field_text == 'Last name':
            name_fields = driver.find_elements(AppiumBy.CLASS_NAME, "android.widget.EditText")
            name_fields[1].send_keys(input_value)
            return True
        input_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((AppiumBy.CLASS_NAME, "android.widget.EditText"))
        )
        if field_text == 'Year':
            input_field.click()
            time.sleep(0.3)
            subprocess.run(["adb", "-s", device_name, "shell", "input", "text", input_value])
        else:
            input_field.send_keys(input_value)
        return True
    except (NoSuchElementException, Exception) as e:
        print(f"Input field '{field_text}' not found: {e}")
        return False
    
def choose_from_dropdown(driver, dropdown_text, resource_Id, value):
    try:
        spinner = driver.find_element(AppiumBy.ANDROID_UIAUTOMATOR, f'new UiSelector().resourceId("{resource_Id}")')
        spinner.click()
        time.sleep(1)

        driver.find_element(AppiumBy.ANDROID_UIAUTOMATOR, f'new UiSelector().text("{value}")').click()
        return True
    except (NoSuchElementException, Exception) as e:
        print(f"Dropdown '{dropdown_text}' or option '{value}' not found: {e}")
        return False

def solve_captcha():
    try:
        subprocess.run([
            "adb", "-s", device_name,
            "shell", "input", "touchscreen", "swipe",
            "540", "1376", "540", "1376", "10000"  # 10 seconds
        ])
    except Exception as e:
        print(f"Error solving captcha: {e}")

def take_action(driver, element_identifier, action):
    try:
        if action == "click":
            success = press_button(driver, element_identifier)
            if success:
                print(f"Clicked on '{action}'")
            else:
                print(f"Failed to click on '{action}'")
        elif action == "type_email":
            email = generate_outlook_email()
            success_email = fill_input_field(driver, element_identifier, email)
            if success_email:
                print(f"Typed email '{email}'")
            else:
                print(f"Failed to type email.")
        elif action == "type_password":
            password = generate_password()
            success_password = fill_input_field(driver, element_identifier, password)
            if success_password:
                print(f"Typed password '{password}'")
            else:
                print(f"Failed to type password.")
        elif action == "type_dob":
            # select month
            random_month = random.randint(1, 12)
            month_text = calendar.month_name[random_month]
            success_month = choose_from_dropdown(driver, 'Month', 'BirthMonthDropdown', month_text)
            if success_month:
                print(f"Typed Month '{month_text}'")
            else:
                print(f"Failed to type Month.")

            # select day
            random_day = random.randint(1, 28)
            success_day = choose_from_dropdown(driver, 'Day', 'BirthDayDropdown', random_day)
            if success_day:
                print(f"Typed Day '{random_day}'")
            else:
                print(f"Failed to type Day.")

            # select year
            year = '2001'
            success_year = fill_input_field(driver, 'Year', year)
            if success_year:
                print(f"Typed Year '{year}'")
            else:
                print(f"Failed to type Year.")
        elif action == "type_fullname":
            first_name = "Ivan"
            last_name = "Lopez"
            success_first = fill_input_field(driver, 'First name', first_name)
            success_last = fill_input_field(driver, 'Last name', last_name)
            if success_first and success_last:
                print(f"Typed full name '{first_name} {last_name}'")
            else:
                print(f"Failed to type full name.")
        elif action == "solve_captcha":
            solve_captcha()
            print("Solved captcha")
    except Exception as e:
        print(f"Error performing action: {e}")

def take_screenshot(device_id, screenshot_number, crop_status_bar=True):

    screenshot_base_dir = os.path.join(screenshots_folder)
    screen_shot_file_name = "screenshot" + ".png"
    screenshot_file = os.path.join(screenshot_base_dir, screen_shot_file_name)
    # remove old if present
    if os.path.exists(screenshot_file):
        try:
            os.remove(screenshot_file)
        except Exception:
            pass

    # run screencap on device
    try:
        cp = subprocess.run(["adb", "-s", device_id, "shell", "screencap", "-p", "/sdcard/screen.png"], capture_output=True, text=True)
        if cp.returncode != 0:
            print(f"adb screencap failed: {cp.stderr.strip()}")
            return None
    except Exception as e:
        print(f"Error running adb screencap: {e}")
        return None

    # pull with retries
    pulled = False
    for attempt in range(3):
        try:
            cp = subprocess.run(["adb", "-s", device_id, "pull", "/sdcard/screen.png", screenshot_file], capture_output=True, text=True)
            if cp.returncode == 0 and os.path.exists(screenshot_file):
                pulled = True
                break
            else:
                print(f"adb pull attempt {attempt+1} failed: {cp.stderr.strip()}")
        except Exception as e:
            print(f"Error pulling screenshot attempt {attempt+1}: {e}")
        time.sleep(1)

    if not pulled:
        print("Failed to pull screenshot from device.")
        return None

    if crop_status_bar:
        try:
            with Image.open(screenshot_file) as img:
                width, height = img.size
                status_bar_height = 120
                cropped_img = img.crop((0, status_bar_height, width, height))
                cropped_img.save(screenshot_file)
        except Exception as e:
            print(f"Error processing screenshot image: {e}")
            return None

    return screenshot_file

completed = False

while not completed:
    text = ""

    while len(text) == 0:
        # use the configured device_name when possible; fall back to emulator id if not set
        device_to_use = globals().get('device_name', 'emulator-5554')
        screenshot_path = take_screenshot(device_to_use, 1, crop_status_bar=True)
        if not screenshot_path:
            # failed to get a screenshot; wait and retry
            time.sleep(2)
            continue
        try:
            text = pytesseract.image_to_string(screenshot_path)
        except Exception as e:
            print(f"OCR error: {e}")
            text = ""
        time.sleep(1)

    if(text):
        try:
            text = analyze_screen_content(text)
            action = json.loads(text)
            if action.get("action") and action.get("element_identifier"):
                take_action(driver, action["element_identifier"], action["action"])
                press_button(driver, "Next")
                if action["element_identifier"] == "Captcha":
                    completed = True
        except Exception as e:
            print(f"Error parsing JSON: {e}")
    time.sleep(2)