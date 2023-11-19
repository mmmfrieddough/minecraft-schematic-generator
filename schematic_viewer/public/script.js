import { renderSchematic, SchematicHandles } from '@enginehub/schematicwebviewer';
import { loadSchematic } from '@enginehub/schematicjs';

import { unzip } from 'gzip-js';
import { decode } from 'nbt-ts';

import io from 'socket.io-client';

const socket = io();

const proxyUrl = 'http://localhost:3000/proxy/';

document.addEventListener('DOMContentLoaded', () => {
  document.addEventListener('dragover', (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
  });

  document.addEventListener('drop', async (event) => {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files.length) {
      await loadSchematicMine(files[0]);
    }
  });

  // Initialize an array to keep track of SchematicHandles instances
  let schematicHandlesList = [];

  async function loadSchematicMine(file) {
    const reader = new FileReader();

    reader.onload = async function (event) {
      const base64String = event.target.result.split(',')[1];

      function parseNbt(nbt) {
          const buff = Buffer.from(nbt, 'base64');
          const deflated = Buffer.from(unzip(buff));
          const data = decode(deflated, {
              unnamed: false,
              useMaps: true,
          });
          return data.value;
      }

      const nbtData = parseNbt(base64String);
      const loadedSchematic = loadSchematic(nbtData);
      const schematicName = loadedSchematic.metadata.Name;
      console.log(schematicName);

      // Set the title at the top of the page
      const titleElement = document.getElementById('schematicTitle');
      titleElement.textContent = schematicName;
  
      // Dispose of any existing schematic handles before loading a new one
      schematicHandlesList.forEach(handle => handle.destroy());
      schematicHandlesList = []; // Clear the array after disposing of the handles
  
      console.log('Loading schematic...');
      try {
        const handle = await renderSchematic(document.querySelector('#schematicRenderer'), base64String, {
          length: 1000,
          height: 1000,
          orbitSpeed: 0.01,
          antialias: true,
          corsBypassUrl: proxyUrl,
          getClientJarUrl: () => 'https://launcher.mojang.com/v1/objects/2e9a3e3107cca00d6bc9c97bf7d149cae163ef21/client.jar',
        });
        console.log(handle)
  
        // Store the new handle in the array
        schematicHandlesList.push(handle);
        console.log('Schematic loaded!');
      } catch (error) {
        console.error('Failed to load schematic:', error);
        alert('Failed to load schematic: ' + error.message);
      }
    };
  
    reader.readAsDataURL(file); // Convert the file to a Data URL (Base64)
  }

  socket.on('file-update', async (data) => {
    console.log('File updated:', data.path);
    try {
      console.log('Fetching file...' + data.path);
      const response = await fetch(data.path);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const fileBlob = await response.blob();
      console.log('Fetched file:', fileBlob);
      // Create a new File object if necessary, or pass the blob directly if loadSchematicMine can handle a Blob
      const fileName = data.path.split('/').pop(); // Extract the file name from the path
      const file = new File([fileBlob], fileName, { type: fileBlob.type });
      console.log('Loading file:', file);
      await loadSchematicMine(file);
    } catch (error) {
      console.error('Failed to fetch new file:', error);
    }
  });
});
