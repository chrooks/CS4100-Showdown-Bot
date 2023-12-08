/**
 * Pokémon Showdown Battle Log Scraper
 *
 * Purpose:
 * This script is designed to scrape battle logs from Pokémon Showdown replays.
 * The scraped data is intended to be used for training machine learning models.
 *
 * Description:
 * The script accesses the Pokémon Showdown replay website, extracts replay links,
 * and downloads the corresponding battle logs in JSON format. The logs are stored
 * in a structured directory format based on the battle format, and each file is named
 * after the battle's unique ID. Additionally, a metadata file is maintained to index
 * all the downloaded logs.
 *
 * Usage:
 * 'npm run crawl'
 * I have this running automatically via a Crontab job every 15 minutes.
 *
 * Libraries Used:
 * - Puppeteer: For controlling a headless browser to access and scrape dynamic content
 *   from the Pokémon Showdown replay website.
 * - Cheerio: For parsing the HTML content obtained from Puppeteer and extracting
 *   necessary information like replay links.
 * - Axios: For making HTTP requests to download the JSON-formatted battle logs.
 * - FS (File System): A Node.js built-in library for file handling, used to save the
 *   downloaded logs and update the metadata file.
 * - Path: A Node.js built-in library for handling file paths.
 */

const puppeteer = require("puppeteer");
const cheerio = require("cheerio");
const axios = require("axios");
const fs = require("fs");
const path = require("path");

async function updateMetadata(format, id, filePath) {
  const metadataPath = path.join(__dirname, "battle-logs", "metadata.json");

  let metadata = {};
  if (fs.existsSync(metadataPath)) {
    metadata = JSON.parse(fs.readFileSync(metadataPath, "utf8"));
  }

  if (!metadata[format]) {
    metadata[format] = [];
  }

  metadata[format].push({ id, filePath });

  fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
}

async function downloadJSON(url, format, id) {
  try {
    const response = await axios.get(url);
    const data = response.data;

    const dirPath = path.join(__dirname, "battle-logs", format);
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
    }

    const filePath = path.join(dirPath, id + ".json");
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
    console.log(`Downloaded: ${filePath}`);

    await updateMetadata(format, id, filePath);
  } catch (error) {
    console.error("Error downloading file:", error);
  }
}

function extractFormatAndId(url) {
  // Updated regex to optionally include query parameters in the URL
  const regex =
    /https:\/\/replay\.pokemonshowdown\.com\/(?:smogtours-)?([a-z0-9]+)-(\d+)(?:\?.*)?\.json/;
  const match = url.match(regex);
  if (match && match.length >= 3) {
    return { format: match[1], id: match[2] };
  } else {
    throw new Error(`Invalid URL format: ${url}`);
  }
}

async function scrapeContent() {
  try {
    const browser = await puppeteer.launch({ headless: "new" });
    const page = await browser.newPage();
    await page.goto("https://replay.pokemonshowdown.com/");
    await page.waitForSelector("#main");

    // Get the HTML of the page
    const htmlContent = await page.evaluate(
      () => document.querySelector("#main").innerHTML
    );

    // Use Cheerio to parse the HTML content
    const $ = cheerio.load(htmlContent);

    // Initialize an empty array to store the links
    let replayLinks = [];

    // Find all anchor tags within 'ul.linklist li' and extract the 'href'
    $("ul.linklist li a").each((index, element) => {
      const href =
        "https://replay.pokemonshowdown.com/" +
        $(element).attr("href") +
        ".json";
      replayLinks.push(href);
    });

    // Close the browser
    await browser.close();

    // Return the array of links
    return replayLinks;
  } catch (error) {
    console.error("Error fetching data:", error);
  }
}

async function main() {
  const replayLinks = await scrapeContent();
  for (const url of replayLinks) {
    const { format, id } = extractFormatAndId(url);
    await downloadJSON(url, format, id);
  }
}

main();
