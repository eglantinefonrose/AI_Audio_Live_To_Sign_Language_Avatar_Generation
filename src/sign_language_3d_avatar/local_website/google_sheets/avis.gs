/**
 * Réception des avis de l'outil Avatar LSFB dans ce tableau Google Sheets.
 *
 * À coller dans Extensions > Apps Script du tableau, puis à déployer comme « Application
 * Web » (voir GOOGLE_SHEETS.md). L'outil envoie chaque avis en POST (JSON) ; le script
 * ajoute une ligne par signe jugé dans l'onglet « Avis ».
 */
const SHEET_NAME = "Avis";
const COLUMNS = ["reçu le", "envoi", "date", "phrase", "glose", "signe_lsfb", "instance", "type",
                 "verdict", "problemes", "commentaire_signe", "commentaire_phrase", "avatar", "sous_titres"];
const MAX_ROWS = 60;       // lignes par envoi (une phrase a au plus 40 gloses)
const MAX_TEXT = 2000;     // caractères par cellule

function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents);
    const rows = (data.lignes || []).slice(0, MAX_ROWS);
    if (!rows.length) return reply({ ok: false, erreur: "aucune ligne" });
    const lock = LockService.getScriptLock();
    lock.waitLock(10000);                     // deux envois en même temps : l'un après l'autre
    try {
      const sheet = getSheet();
      const now = new Date();
      const values = rows.map(r => [now].concat(COLUMNS.slice(1).map(c => cell(r[c]))));
      sheet.getRange(sheet.getLastRow() + 1, 1, values.length, COLUMNS.length).setValues(values);
    } finally {
      lock.releaseLock();
    }
    return reply({ ok: true, lignes: rows.length });
  } catch (err) {
    return reply({ ok: false, erreur: String(err) });
  }
}

/** Ouvrir l'adresse du déploiement dans un navigateur : vérifie qu'il répond. */
function doGet() {
  return reply({ ok: true, service: "avis Avatar LSFB" });
}

/** Texte d'une cellule : tronqué, et jamais interprété comme une formule (=, +, -, @). */
function cell(value) {
  const text = String(value === undefined || value === null ? "" : value).slice(0, MAX_TEXT);
  return /^[=+\-@]/.test(text) ? "'" + text : text;
}

function getSheet() {
  const book = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = book.getSheetByName(SHEET_NAME) || book.insertSheet(SHEET_NAME);
  if (sheet.getLastRow() === 0) {
    sheet.appendRow(COLUMNS);
    sheet.setFrozenRows(1);
  }
  return sheet;
}

function reply(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
