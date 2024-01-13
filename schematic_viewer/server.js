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

const watchDirectory1 = path.join(__dirname, './public/schematics/masked');
const watchDirectory2 = path.join(__dirname, './public/schematics/filled'); // Second directory to watch

console.log(`Watching directories: ${watchDirectory1} and ${watchDirectory2}`);

// Set up file watcher for the first directory
const watcher1 = chokidar.watch(watchDirectory1, {ignored: /^\./, persistent: true});

watcher1
  .on('error', error => console.log(`Watcher1 error: ${error}`))
  .on('ready', () => console.log(`Initial scan complete. Ready for changes in ${watchDirectory1}`))
  .on('add', filePath => handleFileUpdate(filePath, 'file-update-1')) // Emit a specific event for directory 1
  .on('change', filePath => handleFileUpdate(filePath, 'file-update-1'));

// Set up file watcher for the second directory
const watcher2 = chokidar.watch(watchDirectory2, {ignored: /^\./, persistent: true});

watcher2
  .on('error', error => console.log(`Watcher2 error: ${error}`))
  .on('ready', () => console.log(`Initial scan complete. Ready for changes in ${watchDirectory2}`))
  .on('add', filePath => handleFileUpdate(filePath, 'file-update-2')) // Emit a specific event for directory 2
  .on('change', filePath => handleFileUpdate(filePath, 'file-update-2'));

// Handle file updates and emit events to the browser
const handleFileUpdate = (filePath, event) => {
  console.log(`File ${filePath} has been added or changed`);
  const filename = path.basename(filePath);
  const browserPath = event === 'file-update-1' ? `/schematics/masked/${filename}` : `/schematics/filled/${filename}`;
  io.emit(event, { path: browserPath });
};

console.log('Watching directories for changes...');
