// Vercel Web Analytics (mesure d'audience anonyme, sans cookie tiers) : file d'attente
// window.va de la documentation Vercel, puis le script servi par Vercel lui-même.
// Rien n'est chargé en aperçu local, où le script n'existe pas. Les valeurs saisies dans
// le formulaire ne sont jamais envoyées : seule la visite de la page est comptée.
"use strict";

window.va = window.va || function () {
  (window.vaq = window.vaq || []).push(arguments);
};

if (!["localhost", "127.0.0.1", "[::1]"].includes(window.location.hostname)) {
  const script = document.createElement("script");
  script.defer = true;
  script.src = "/_vercel/insights/script.js";
  document.head.append(script);
}
