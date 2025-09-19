from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import time
import csv

#NEED TO UPDATE AS IT DOES NOT WORK CONSISTENTLY FOR THE LAST THREE EARNINGS CALLS

def scrape_all_transcripts():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Set to True for headless mode
        page = browser.new_page()
        page.goto('https://www.fool.com/quote/nasdaq/nvda/#quote-earnings-transcripts')

        # Keep clicking 'Load more' until it's gone
        while True:
            try:
                load_more = page.locator('button', has_text='View More NVDA Earnings Transcripts')
                if load_more.is_visible():
                    load_more.click()
                    time.sleep(2)
                else:
                    break
            except:
                break

        # Get the final HTML after all transcripts are loaded
        html = page.content()
        browser.close()

    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    transcripts = [
        h3 for h3 in soup.find_all(
        'h3', 
        class_ = 'mb-6 font-medium hover-target-cyan-700 mt-2px text-h5',
        string = lambda text: text and "Earnings Call Transcript" in text
        )
        if "earnings call transcript" in h3.get_text(strip=True).lower()
    ]

    results = []
    for t in transcripts:
        title = t.get_text(strip=True)
        # Publication date
        date_publication_div = t.find_next('div', class_=['text-sm', 'text-gray-800', 'mb-2px'])
        if date_publication_div:
            text = date_publication_div.get_text(strip=True)
            publication_date = text.split("|")[-1].strip() if "|" in text else text
        else:
            publication_date = "N/A"

        # Quarter End date
        date_end_quarter_div = t.find_next('p', class_ = 'text-gray-800 mb-0px')
        if date_end_quarter_div:
            text = date_end_quarter_div.get_text(strip=True)
            quarter_end_date = text.split("ending")[-1].strip() if "ending" in text else text
        else:
            quarter_end_date = "N/A"

        # Link for Earnings Call Transcript (parent <a>)
        link_tag = t.parent if t.parent.name == 'a' else None
        url = link_tag['href'] if link_tag and link_tag.has_attr('href') else "N/A"
        full_url = f"https://www.fool.com{url}"

        print(f"Title: {title}\nPublication Date: {publication_date}\nQuarter End Date: {quarter_end_date}\nURL: {full_url}\n")

        # Add results to a list as a dictionary
        results.append({
            "Title": title,
            "Publication Date": publication_date,
            "Quarter End Date": quarter_end_date,
            "URL": full_url
        })

    #Write results to CSV
    with open('nvidia_earnings_calls.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Title", "Publication Date", "Quarter End Date", "URL"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("Saved to nvidia_earnings_calls.csv")

if __name__ == "__main__":
    scrape_all_transcripts()