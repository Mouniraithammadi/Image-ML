
import os
import time

import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# The URL of the website to scrape
url = "https://www.iconfinder.com/search?q=emoji&price=free&style=flat"

# The folder to store the downloaded images
folder = "images"

# Create the folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

# Initialize a WebDriver instance
driver = webdriver.Chrome()  # Replace with the path to your Chrome webdriver

# Navigate to the URL
driver.get(url)

# Scroll to the bottom of the page to load more content
while len(driver.find_elements(By.TAG_NAME, "img")) < 1000:
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
    print(len(driver.find_elements(By.TAG_NAME, "img")))
    time.sleep(2)  # Adjust sleep time as needed

# Parse the HTML content
soup = BeautifulSoup(driver.page_source, "html.parser")

# Find all <img> tags in the HTML content
img_tags = soup.find_all("img")
i = 0
# Download each image and save it to the folder
for img_tag in img_tags:
    # Get the URL of the image
    img_url = img_tag["src"]
    if img_url.endswith(".png"):
        # Send a request to download the image
        response = requests.get(img_url)

        # Save the image to a file in the folder
        filename = os.path.join(folder, os.path.basename(img_url))
        with open(filename, "wb") as f:
            f.write(response.content)
            print(i)
            i += 1

# Close the WebDriver
driver.quit()
