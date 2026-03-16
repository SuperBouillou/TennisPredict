const CACHE = 'tennis-v1';
const STATIC = ['/static/app.css'];
self.addEventListener('install', e =>
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(STATIC)))
);
self.addEventListener('fetch', e => {
  if (e.request.url.includes('/static/')) {
    e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
  }
});
