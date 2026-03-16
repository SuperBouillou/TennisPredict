import requests

sources = [
    ("Sackmann GitHub", [
        f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{y}.csv"
        for y in [2025, 2026]
    ]),
    ("tennis-data.co.uk", [
        f"http://www.tennis-data.co.uk/{y}/{y}.xlsx"
        for y in [2025, 2026]
    ]),
]

for source, urls in sources:
    print(f"\n── {source} ──")
    for url in urls:
        try:
            r = requests.head(url, timeout=10)
            size = r.headers.get("Content-Length", "?")
            status = "OK" if r.status_code == 200 else "absent"
            print(f"  {url.split('/')[-1]}: {r.status_code} - {status} ({size} bytes)")
        except Exception as e:
            print(f"  Erreur : {e}")
