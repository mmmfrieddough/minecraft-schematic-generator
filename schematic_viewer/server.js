const express = require('express');
const corsAnywhere = require('cors-anywhere');
const path = require('path');
const Bundler = require('parcel-bundler');
const socketIo = require('socket.io');
const chokidar = require('chokidar');
const http = require('http');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);
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

app.use('/schematics', express.static(path.join(__dirname, 'public/schematics')));

// Point to the entry file of your application
const entryFiles = path.join(__dirname, './public/index.html');
const options = {}; // Parcel options, if you have any

// Initialize a new parcel bundler
const bundler = new Bundler(entryFiles, options);

// Let express use the bundler middleware
app.use(bundler.middleware());

server.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});

const watchDirectory = path.join(__dirname, './public/schematics');
console.log(`Watching directory: ${watchDirectory}`);

// Set up file watcher
const watcher = chokidar.watch(watchDirectory, {ignored: /^\./, persistent: true});

watcher
  .on('error', error => console.log(`Watcher error: ${error}`))
  .on('ready', () => console.log(`Initial scan complete. Ready for changes in ${watchDirectory}`));

const handleFileUpdate = (filePath) => {
  console.log(`File ${filePath} has been added or changed`);
  const filename = path.basename(filePath);
  const browserPath = `/schematics/${filename}`;
  io.emit('file-update', { path: browserPath });
};
  
watcher
  .on('add', handleFileUpdate)

console.log('Watching directory for changes...');
