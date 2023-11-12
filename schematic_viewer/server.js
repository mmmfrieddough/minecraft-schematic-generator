const express = require('express');
const corsAnywhere = require('cors-anywhere');
const path = require('path');
const Bundler = require('parcel-bundler');
const app = express();
const PORT = 3000;

// Create a CORS Anywhere proxy
  const proxy = corsAnywhere.createServer({
  originWhitelist: [], // Allow all origins
  requireHeader: [], // Do not require any headers.
  removeHeaders: [] // Strip cookies for privacy
});

// Use the proxy on the /proxy route
app.use('/proxy', (req, res) => {
  req.url = req.url.replace('/proxy/', '/'); // Strip '/proxy' from the front of the URL
  proxy.emit('request', req, res);
});

// Point to the entry file of your application
const entryFiles = path.join(__dirname, './public/index.html');
const options = {}; // Parcel options, if you have any

// Initialize a new parcel bundler
const bundler = new Bundler(entryFiles, options);

// Let express use the bundler middleware
app.use(bundler.middleware());

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});