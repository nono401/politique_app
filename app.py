
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --- Titre du site ---
st.title("🗳️ Carte Politique Interactive")
st.write("Répondez aux questions pour connaître votre position politique selon l'analyse du CEVIPOF.")

# --- Chargement des modèles ---
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("pca.pkl", "rb") as f:
        pca = pickle.load(f)
    with open("kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("df_pca.pkl", "rb") as f:
        df_pca = pickle.load(f)
    st.success("Modèles chargés avec succès ✔️")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement des modèles : {e}")
    st.stop()

# --- Questions ---
questions_text = {
    "TAX_DROIT_SUCCESSION": "Il faut taxer plus fortement les droits de succession.",
    "ENCADR_LOYER": "L'État doit prendre des mesures pour encadrer les loyers.",
    "PRENDRE_RICHE": "Pour rétablir la justice sociale, il faut prendre aux riches pour donner aux pauvres.",
    "REV_UNIVERSEL": "Il faut instaurer un revenu universel pour tous les jeunes.",
    "AUGM_SALAIRE": "L'État doit forcer les entreprises à augmenter les salaires.",
    "ECOLE_DISCIPLINE": "L'école doit donner le sens de la discipline et de l'effort.",
    "EXCUSE_COLONISATION": "La France doit s'excuser pour la colonisation.",
    "PMA_BON": "La PMA est une bonne chose pour les femmes seules ou homosexuelles.",
    "INVEST_SERV_PUBLIC": "L'État doit investir massivement dans les services publics.",
    "PUNIR_DELINQU": "Il faut punir plus durement les délinquants.",
    "REDUIR_DROIT_MANIF": "Il faut réduire le droit de manifester.",
    "LICENCIEMENT_FACIL": "Les patrons doivent pouvoir licencier plus facilement.",
    "REDUIR_FONCTIONNAIRE": "Il faut réduire le nombre de fonctionnaires.",
    "BAISSE_CHARGE_ENTR": "Il faut baisser les charges des entreprises.",
    "SORTIR_OTAN": "La France doit sortir de l'OTAN.",
    "AVANTAG_UE": "La France tire plus d'avantages que d'inconvénients de l'UE.",
    "UE_PROTEG_MONDIAL": "L’UE protège des effets négatifs de la mondialisation.",
    "REDUIR_NUCLEAIRE": "Il faut réduire la part du nucléaire.",
    "CHANG_ECONOM_MARCHE": "La transition écologique nécessite de revoir le marché économique."
}

questions = list(questions_text.keys())

# --- Collecte des réponses ---
st.subheader("📋 Répondez aux questions")

reponses = {q: st.slider(questions_text[q], -2, 2, 0) for q in questions}

# --- Bouton analyser ---
if st.button("Analyser ma position politique"):

    # Convertir les réponses en DataFrame
    user = pd.DataFrame([reponses])

    # PCA
    user_scaled = scaler.transform(user)
    user_pca = pca.transform(user_scaled)[0]

    # Cluster
    user_cluster = kmeans.predict([user_pca])[0]

    # --- Candidat le plus proche ---
    df_pca["distance"] = np.sqrt(
        (df_pca["PC1"] - user_pca[0])**2 +
        (df_pca["PC2"] - user_pca[1])**2
    )
    closest = df_pca.loc[df_pca["distance"].idxmin()]
    closest_name = closest["CANDIDAT"]

    st.success(f"🎯 Vous êtes politiquement le plus proche de : **{closest_name}**")

    # --- Description du cluster ---
    cluster_desc = {
        0: "🟩 Gauche économique / écologiste / progressiste.",
        1: "🟦 Centre / libéral modéré.",
        2: "🟥 Droite libérale / conservatrice / souverainiste."
    }

    st.info(f"🧭 **Interprétation politique :** {cluster_desc[user_cluster]}")

    # --- Affichage des coordonnées ---
    st.write("### 📌 Vos coordonnées dans l’espace politique :")
    st.write(f"**PC1 (économique)** : `{user_pca[0]:.3f}`")
    st.write(f"**PC2 (social)** : `{user_pca[1]:.3f}`")
    st.write(f"**Cluster** : `{user_cluster}`")

    # --- Graphique ---
    st.subheader("🗺️ Votre position sur la carte politique")

    fig, ax = plt.subplots(figsize=(8, 8))

    for c in sorted(df_pca["cluster"].unique()):
        subset = df_pca[df_pca["cluster"] == c]
        ax.scatter(subset["PC1"], subset["PC2"], s=80, label=f"Cluster {c}")
        for _, row in subset.iterrows():
            ax.text(row["PC1"]+0.05, row["PC2"]+0.05, row["CANDIDAT"], fontsize=9)

    ax.scatter(user_pca[0], user_pca[1], c="red", s=200, edgecolors="black")
    ax.text(user_pca[0]+0.1, user_pca[1]+0.1, "Vous", color="red", fontsize=12, fontweight="bold")

    ax.axhline(0, color="black")
    ax.axvline(0, color="black")
    ax.set_xlabel("Interventionnisme étatique")
    ax.set_ylabel("Libéralisme économique")
    ax.legend()

    st.pyplot(fig)
