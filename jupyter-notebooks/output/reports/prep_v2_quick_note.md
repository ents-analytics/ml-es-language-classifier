# Prep v2 quick note

- Input: `data/interim/billboard_chart_data_interim1.csv`
- Output: `data/interim/billboard_chart_data_prepped_v2.parquet`

## Added columns

- `song_raw` / `artist_raw`: preserved originals

- `song_base`: lowercased, accents stripped, parentheses removed, safe '&' -> 'and'

- `paren_tag`: first parenthetical content (if any)

- `version_type`: coarse type derived from `paren_tag` (live/remix/acoustic/edit/language/other/none)

- `artist_canon`: canonical display (handles 'Smiths, The' -> 'the smiths')

- `artist_key`: join key (drops leading 'the ')


## Rationale

- Use `song_base` + `artist_key` for robust joins (Spotify/lyrics).

- Use `paren_tag` / `version_type` to disambiguate Spotify candidates (e.g., prefer live vs studio).

- Keep originals for auditability and reporting.
