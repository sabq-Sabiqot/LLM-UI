const express = require('express');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files
app.use(express.static(path.join(__dirname, 'dist')));

// Loading page middleware (simulate loading for demonstration)
app.get('/loading', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'loading.html'));
});

// Error page route (simulate error)
app.get('/error', (req, res) => {
  res.status(500).sendFile(path.join(__dirname, 'dist', 'error.html'));
});

// 404 handler (must be last)
app.use((req, res) => {
  res.status(404).sendFile(path.join(__dirname, 'dist', '404.html'));
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
