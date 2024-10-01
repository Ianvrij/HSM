import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from io import BytesIO

# Pagina-configuratie instellen
st.set_page_config(page_title="Bedrijfsconfigurator", layout="wide")

# Historische data
historische_data = {
    "maand": ["jan-23", "feb-23", "mrt-23", "apr-23", "mei-23", "jun-23", "jul-23", "aug-23", "sep-23", "okt-23", "nov-23", "dec-23", "jan-24", "feb-24", "mrt-24", "apr-24", "mei-24", "jun-24", "jul-24", "aug-24"],
    "laadpalen": [14, 10, 11, 21, 15, 12, 10, 17, 10, 22, 5, 8, 20, 24, 23, 17, 31, 22, 20, 15],
    "zonnepanelen": [0, 0, 0, 0, 1, 0, 6, 4, 3, 3, 0, 1, 0, 1, 0, 1, 1, 2, 0, 0],
    "omzet": [34972.56, 22937.56, 37377.71, 49327.57, 48314.05, 27511.87, 73155.08, 71720.10, 59371.58, 66803.77, 22026.94, 23133.60, 40926.10, 71023.41, 52654.90, 44436.37, 74361.93, 60974.46, 42048.34, 33335.69],
    "kostprijs": [-13490.24, -10655.51, -11404.87, -15546.61, -20212.57, -10012.42, -15405.87, -24661.99, -34189.33, -33250.14, -3023.21, -6984.25, -22226.84, -40685.79, -23198.18, -18144.26, -31619.83, -16394.96, -19376.19, -14653.75],
    "brutomarge": [21482.32, 12282.06, 25972.85, 33780.96, 28101.48, 17499.45, 57749.21, 47058.12, 25182.25, 33553.63, 19003.73, 16149.35, 18699.26, 30337.62, 29456.72, 26292.11, 42742.10, 44579.50, 22672.15, 18681.94],
    "omzet_laadpalen": [34972.56, 22937.56, 37377.71, 49327.57, 39464.04, 27511.87, 22764.99, 41804.31, 26672.93, 47580.16, 22026.94, 16282.03, 40926.10, 60823.41, 52654.90, 36150.27, 68943.37, 43275.11, 42048.34, 33335.69],
    "kostprijs_laadpalen": [-13490.24, -10655.51, -11404.87, -15546.61, -12272.30, -10012.42, -7944.58, -17384.96, -21397.47, -25936.72, -3023.21, -3578.20, -22226.84, -37084.60, -23198.18, -15212.16, -29873.07, -11550.28, -19376.19, -14653.75],
    "brutomarge_laadpalen": [21482.32, 12282.06, 25972.85, 33780.96, 27191.74, 17499.45, 14820.41, 24419.36, 5275.46, 21643.44, 19003.73, 12703.83, 18699.26, 23738.81, 29456.72, 20938.11, 39070.30, 31724.83, 22672.15, 18681.94],
    "omzet_zonnepanelen": [0, 0, 0, 0, 8850.01, 0, 50390.09, 29915.79, 32698.65, 19223.61, 0, 6851.57, 0, 10200.00, 0, 8286.10, 5418.56, 17699.35, 0, 0],
    "kostprijs_zonnepanelen": [0, 0, 0, 0, -7940.27, 0, -7461.29, -7277.03, -12791.86, -7313.42, 0, -3406.05, 0, -3601.19, 0, -2932.10, -1746.76, -4844.68, 0, 0],
    "brutomarge_zonnepanelen": [0, 0, 0, 0, 909.74, 0, 42928.80, 22638.76, 19906.79, 11910.19, 0, 3445.52, 0, 6598.81, 0, 5354.00, 3671.80, 12854.67, 0, 0],
    "personeelskosten": [-16315.00] * 20,
    "it_kosten": [-311.29, -311.29, -311.29, -311.29, -284.03, -391.83, -391.83, -391.83, -284.03, -391.83, -391.83, -391.83, -311.29, -311.29, -311.29, -311.29, -284.03, -391.83, -391.83, -391.83],
    "solar_kosten": [-704.50] * 20,
    "contributie_kosten": [-154.74] * 20,
    "afschrijving_kosten": [-3631.45] * 20,
    "resultaat": [365.34, -8834.93, 4855.87, 12663.98, 7011.76, -3415.60, 36834.16, 26143.06, 4375.00, 12638.58, -1911.32, -4765.70, -2417.72, 9220.64, 8339.74, 5175.13, 21834.67, 23664.45, 1757.10, -2233.11]
}

# Data inladen in DataFrame
df = pd.DataFrame(historische_data)

# Data uit sessie laden als beschikbaar
if "data" not in st.session_state:
    st.session_state.data = df.copy()
else:
    df = st.session_state.data.copy()

if "specificaties" not in st.session_state:
    st.session_state.specificaties = {}
else:
    specificaties = st.session_state.specificaties.copy()

# Maanden in het Nederlands
dutch_months = {
    "jan": 1, "feb": 2, "mrt": 3, "apr": 4, "mei": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "okt": 10, "nov": 11, "dec": 12
}

# Functie om maand te parsen
def parse_month(month_str):
    month_abbr, year = month_str.split('-')
    month_num = dutch_months[month_abbr]
    year_num = 2000 + int(year)
    return datetime(year_num, month_num, 1)

def get_next_month(month_str):
    current_month_dt = parse_month(month_str)
    year = current_month_dt.year + (current_month_dt.month // 12)
    month = current_month_dt.month % 12 + 1
    next_month_dt = datetime(year, month, 1)
    month_abbr_reverse = {v: k for k, v in dutch_months.items()}
    next_month_abbr = month_abbr_reverse[next_month_dt.month]
    next_month_str = f"{next_month_abbr}-{str(next_month_dt.year)[-2:]}"
    return next_month_str

def parse_month_series(month_series):
    def parse_single_month(month_str):
        month_abbr, year = month_str.split('-')
        month_num = dutch_months[month_abbr]
        year_num = 2000 + int(year)
        return datetime(year_num, month_num, 1)
    return month_series.apply(parse_single_month)

# Data sorteren op maand
df_sorted = df.sort_values('maand', key=parse_month_series).reset_index(drop=True)

# Gemiddelden instellen naar vaste waarden
gemiddelde_marge_per_laadpalen = 57.51
gemiddelde_marge_per_zonnepaneel = 68.70

# Functie om gegevens te berekenen op basis van invoer
def bereken_gegevens(aantal_laadpalen, aantal_zonnepanelen, marge_laadpalen, marge_zonnepanelen,
                     aantal_installeurs, aantal_verkopers, fulltime_verkopers, overige_personeelskosten,
                     marketing_budget, maand,
                     gemiddelde_omzet_per_laadpaal, gemiddelde_marge_per_laadpalen,
                     gemiddelde_omzet_per_zonnepaneel, gemiddelde_marge_per_zonnepaneel):
    # Omzet en marge berekeningen
    omzet_laadpalen = aantal_laadpalen * gemiddelde_omzet_per_laadpaal
    marge_laadpalen_bedrag = aantal_laadpalen * gemiddelde_omzet_per_laadpaal * (marge_laadpalen / 100)

    omzet_zonnepanelen = aantal_zonnepanelen * gemiddelde_omzet_per_zonnepaneel
    marge_zonnepanelen_bedrag = aantal_zonnepanelen * gemiddelde_omzet_per_zonnepaneel * (marge_zonnepanelen / 100)

    totale_omzet = omzet_laadpalen + omzet_zonnepanelen
    totale_marge = marge_laadpalen_bedrag + marge_zonnepanelen_bedrag

    # Personeelskosten
    fulltime_installeurs_kosten = aantal_installeurs * 4000
    parttime_verkopers_kosten = aantal_verkopers * (20 / 40) * 3000
    fulltime_verkoper_kosten = fulltime_verkopers * 3000

    personeelskosten = fulltime_installeurs_kosten + parttime_verkopers_kosten + fulltime_verkoper_kosten + overige_personeelskosten

    # Vaste kosten
    it_kosten = -df["it_kosten"].mean()
    solar_kosten = -df["solar_kosten"].mean()
    contributie_kosten = -df["contributie_kosten"].mean()

    # Afschrijving blijft constant
    afschrijving_kosten = +3631.45

    vaste_kosten = it_kosten + solar_kosten + contributie_kosten + afschrijving_kosten
    totale_kosten = personeelskosten + vaste_kosten + marketing_budget

    resultaat = totale_marge - totale_kosten

    nieuwe_data = {
        "maand": maand,
        "laadpalen": aantal_laadpalen,
        "zonnepanelen": aantal_zonnepanelen,
        "omzet": totale_omzet,
        "kostprijs": -(totale_omzet - totale_marge),
        "brutomarge": totale_marge,
        "omzet_laadpalen": omzet_laadpalen,
        "kostprijs_laadpalen": -(omzet_laadpalen - marge_laadpalen_bedrag),
        "brutomarge_laadpalen": marge_laadpalen_bedrag,
        "omzet_zonnepanelen": omzet_zonnepanelen,
        "kostprijs_zonnepanelen": -(omzet_zonnepanelen - marge_zonnepanelen_bedrag),
        "brutomarge_zonnepanelen": marge_zonnepanelen_bedrag,
        "personeelskosten": -personeelskosten,
        "it_kosten": -it_kosten,
        "solar_kosten": -solar_kosten,
        "contributie_kosten": -contributie_kosten,
        "afschrijving_kosten": -afschrijving_kosten,
        "resultaat": resultaat
    }

    specificatie_nieuwe_maand = {
        "Personeelskosten": personeelskosten,
        "IT kosten": it_kosten,
        "Solar kosten": solar_kosten,
        "Contributie installatiebedrijf": contributie_kosten,
        "Afschrijvingen": afschrijving_kosten,
    }

    return nieuwe_data, specificatie_nieuwe_maand

# Hoofdtitel van de applicatie
st.title("Bedrijfsconfigurator")

# Functie om invoer bij te werken wanneer de maand verandert
def update_inputs():
    selected_month = st.session_state['selected_month']
    if selected_month and selected_month != "Selecteer een maand":
        maand_data = df[df['maand'] == selected_month].iloc[0]
        st.session_state['aantal_laadpalen'] = int(maand_data['laadpalen'])
        st.session_state['aantal_zonnepanelen'] = int(maand_data['zonnepanelen'])
        # Removed dynamic margin updates
    else:
        st.session_state['aantal_laadpalen'] = 0
        st.session_state['aantal_zonnepanelen'] = 0
        # Removed dynamic margin updates

# Sidebar voor invoerparameters
st.sidebar.header("Invoerparameters")

# Maandselectie
maand_options = df['maand'].unique().tolist()
latest_month_str = sorted(maand_options, key=lambda x: parse_month(x))[-1]
next_month = get_next_month(latest_month_str)
if next_month not in maand_options:
    maand_options.append(next_month)
maand_options = sorted(set(maand_options), key=lambda x: parse_month(x))

maand_to_adjust = st.sidebar.selectbox("Selecteer de maand om aan te passen", maand_options, index=maand_options.index(next_month), key='maand_to_adjust')

# Verkoopgegevens
st.sidebar.subheader("Verkoopgegevens")
aantal_laadpalen = st.sidebar.number_input(
    "Aantal verkochte laadpalen",
    min_value=0,
    max_value=100,
    value=st.session_state.get('aantal_laadpalen', 0),
    key='aantal_laadpalen'
)
aantal_zonnepanelen = st.sidebar.number_input(
    "Aantal verkochte zonnepanelen",
    min_value=0,
    max_value=50,
    value=st.session_state.get('aantal_zonnepanelen', 0),
    key='aantal_zonnepanelen'
)
# Set default margins to average values
marge_laadpalen = st.sidebar.number_input(
    "Marge Laadpalen (%)",
    min_value=0.0,
    max_value=100.0,
    value=st.session_state.get('marge_laadpalen', 57.51),
    key='marge_laadpalen'
)
marge_zonnepanelen = st.sidebar.number_input(
    "Marge Zonnepanelen (%)",
    min_value=0.0,
    max_value=100.0,
    value=st.session_state.get('marge_zonnepanelen', 68.70),
    key='marge_zonnepanelen'
)

# Personeelsgegevens
st.sidebar.subheader("Personeelsgegevens")
aantal_installeurs = st.sidebar.number_input(
    "Aantal fulltime installateurs",
    min_value=1,
    max_value=10,
    value=st.session_state.get('aantal_installeurs', 2),
    key='aantal_installeurs'
)
aantal_verkopers = st.sidebar.number_input(
    "Aantal parttime verkopers",
    min_value=0,
    max_value=10,
    value=st.session_state.get('aantal_verkopers', 2),
    key='aantal_verkopers'
)
fulltime_verkopers = st.sidebar.number_input(
    "Aantal fulltime verkopers",
    min_value=0,
    max_value=10,
    value=st.session_state.get('fulltime_verkopers', 0),
    key='fulltime_verkopers'
)
overige_personeelskosten = st.sidebar.number_input(
    "Overige personeelskosten (€)",
    value=st.session_state.get('overige_personeelskosten', 4815.00),
    step=100.00,
    key='overige_personeelskosten'
)

# Marketing
st.sidebar.subheader("Marketing")
marketing_budget = st.sidebar.number_input(
    "Marketing Budget (€)",
    min_value=0,
    max_value=50000,
    value=st.session_state.get('marketing_budget', 5000),
    key='marketing_budget'
)

# Invoeren-knop
if st.sidebar.button("Invoeren", key='invoer_knop'):
    # Bestaande data voor de maand verwijderen
    df = df[df['maand'] != maand_to_adjust]

    # Gegevens voor de nieuwe maand berekenen
    nieuwe_data, specificatie_nieuwe_maand = bereken_gegevens(
        aantal_laadpalen,
        aantal_zonnepanelen,
        marge_laadpalen,
        marge_zonnepanelen,
        aantal_installeurs,
        aantal_verkopers,
        fulltime_verkopers,
        overige_personeelskosten,
        marketing_budget,
        maand_to_adjust,
        gemiddelde_omzet_per_laadpaal= df["omzet_laadpalen"].sum() / df["laadpalen"].sum() if df["laadpalen"].sum() !=0 else 0,
        gemiddelde_marge_per_laadpalen= gemiddelde_marge_per_laadpalen,
        gemiddelde_omzet_per_zonnepaneel= df["omzet_zonnepanelen"].replace(0, np.nan).dropna().sum() / df["zonnepanelen"].replace(0, np.nan).dropna().sum() if df["zonnepanelen"].replace(0, np.nan).dropna().sum() !=0 else 0,
        gemiddelde_marge_per_zonnepaneel= gemiddelde_marge_per_zonnepaneel
    )

    # DataFrame bijwerken
    df = pd.concat([df, pd.DataFrame([nieuwe_data])], ignore_index=True)
    df = df.sort_values('maand', key=lambda x: x.apply(parse_month)).reset_index(drop=True)
    st.session_state.data = df

    # Specificaties bijwerken
    st.session_state.specificaties[maand_to_adjust] = specificatie_nieuwe_maand
    specificaties = st.session_state.specificaties

    st.success(f"Gegevens voor {maand_to_adjust} zijn bijgewerkt.")

# Layout met tabs
tab1, tab2 = st.tabs(["Resultaten", "Visualisaties"])

with tab1:
    st.header("Resultaten")

    # Maandselectie voor resultaten
    selected_month = st.selectbox(
        "Selecteer een maand",
        ["Selecteer een maand"] + sorted(df['maand'].unique().tolist(), key=lambda x: parse_month(x)),
        key='selected_month',
        on_change=update_inputs
    )

    if selected_month and selected_month != "Selecteer een maand":
        # Data filteren voor geselecteerde maand
        maand_data = df[df['maand'] == selected_month].iloc[0]

        # Resultaten tonen
        with st.container():
            st.markdown("---")  # Horizontale lijn voor visuele scheiding

            # Gebruik een multikolom layout voor een betere presentatie
            col1, col2, col3, col4 = st.columns(4)

            totale_personen = (
                aantal_installeurs +
                aantal_verkopers +
                fulltime_verkopers +
                1  # Inclusief administratief personeel
            )

            with col1:
                st.markdown("#### Omzet")
                st.metric("Totale Omzet", f"€ {maand_data['omzet']:,.2f}")

            with col2:
                st.markdown("#### Marge")
                st.metric("Totale Marge", f"€ {maand_data['brutomarge']:,.2f}")

            with col3:
                st.markdown("#### Resultaat")
                st.metric("Resultaat", f"€ {maand_data['resultaat']:,.2f}")

            with col4:
                st.markdown("#### Kosten")
                st.metric("Personeelskosten", f"€ {-maand_data['personeelskosten']:,.2f}")

            st.markdown("---")

            # Specificaties weergeven als DataFrame
            specificatie_data = st.session_state.specificaties.get(selected_month, {})
            if specificatie_data:
                st.markdown("#### Specificaties")
                spec_df = pd.DataFrame.from_dict(specificatie_data, orient='index', columns=['Bedrag (€)'])
                spec_df['Bedrag (€)'] = spec_df['Bedrag (€)'].apply(lambda x: f"€ {x:,.2f}")
                st.table(spec_df)

            st.markdown("---")

            # Kostenoverzicht weergeven
            st.markdown("#### Kostenoverzicht")
            kosten_data = {
                "Categorie": ["IT kosten", "Solar kosten", "Contributie installatiebedrijf", "Afschrijving vervoersmiddelen", "Marketingkosten"],
                "Bedrag (€)": [
                    f"€ {-maand_data['it_kosten']:,.2f}",
                    f"€ {-maand_data['solar_kosten']:,.2f}",
                    f"€ {-maand_data['contributie_kosten']:,.2f}",
                    f"€ {-maand_data['afschrijving_kosten']:,.2f}",
                    f"€ {marketing_budget:,.2f}"
                ]
            }
            kosten_df = pd.DataFrame(kosten_data)
            st.table(kosten_df)

            st.markdown("---")

            # Alle detailgegevens voor alle maanden weergeven
            st.markdown("#### Detailgegevens voor Alle Maanden")
            all_months_data = df_sorted.copy()
            for col in all_months_data.columns:
                if all_months_data[col].dtype == 'float':
                    all_months_data[col] = all_months_data[col].apply(lambda x: f"€ {x:,.2f}")
            st.dataframe(all_months_data)

with tab2:
    st.header("Visualisaties")

    if 'selected_month' in st.session_state and st.session_state['selected_month'] != "Selecteer een maand":
        maand_data = df[df['maand'] == st.session_state['selected_month']].iloc[0]
        totale_personen = (
            aantal_installeurs +
            aantal_verkopers +
            fulltime_verkopers +
            1
        )

        with st.container():
            col1, col2, col3 = st.columns(3)

            with col1:
                fig1 = px.bar(
                    x=["Omzet", "Marge", "Resultaat"],
                    y=[maand_data['omzet'], maand_data['brutomarge'], maand_data['resultaat']],
                    labels={'x': 'Categorie', 'y': 'Bedrag in €'},
                    title="Financieel Overzicht",
                    color_discrete_sequence=["#636EFA"]  # Toevoegen van kleur
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = px.bar(
                    x=["Omzet Laadpalen", "Marge Laadpalen"],
                    y=[maand_data['omzet_laadpalen'], maand_data['brutomarge_laadpalen']],
                    labels={'x': 'Categorie', 'y': 'Bedrag in €'},
                    title="Laadpalen Omzet en Marge",
                    color_discrete_sequence=["#EF553B"]
                )
                st.plotly_chart(fig2, use_container_width=True)

            with col3:
                fig3 = px.bar(
                    x=["Omzet Zonnepanelen", "Marge Zonnepanelen"],
                    y=[maand_data['omzet_zonnepanelen'], maand_data['brutomarge_zonnepanelen']],
                    labels={'x': 'Categorie', 'y': 'Bedrag in €'},
                    title="Zonnepanelen Omzet en Marge",
                    color_discrete_sequence=["#00CC96"]
                )
                st.plotly_chart(fig3, use_container_width=True)

    # Trends en verhoudingen
    st.markdown("### Trends en Verhoudingen")

    with st.container():
        df_sorted = df.sort_values('maand', key=lambda x: x.apply(parse_month))
        fig4 = px.line(
            df_sorted,
            x="maand",
            y=["omzet", "kostprijs", "resultaat"],
            labels={'value': 'Bedrag in €', 'variable': 'Categorie'},
            title="Omzet, Kosten en Winst Over Tijd",
            markers=True
        )
        st.plotly_chart(fig4, use_container_width=True)

        fig5 = px.line(
            df_sorted,
            x="maand",
            y=["brutomarge_laadpalen", "brutomarge_zonnepanelen"],
            labels={'value': 'Bedrag in €', 'variable': 'Categorie'},
            title="Marge Over Tijd per Product",
            markers=True
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Downloadbare CSV
    st.markdown("### Download Data")
    csv_data = df_sorted.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv_data, file_name="bedrijfsconfigurator_data.csv", mime="text/csv")

# PDF Rapport Generatie

# Helper function to parse Dutch month strings
def parse_dutch_month(month_str):
    dutch_months = {
        "jan": 1, "feb": 2, "mrt": 3, "apr": 4, "mei": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "okt": 10, "nov": 11, "dec": 12
    }
    month_abbr, year = month_str.split('-')
    month_num = dutch_months[month_abbr]
    year_num = 2000 + int(year)  # Assuming the format 'yy' for years in the 2000s
    return datetime(year_num, month_num, 1)

# Custom PDF class
class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", size=8)
        self.cell(0, 10, f"Pagina {self.page_no()}", 0, 0, "C")

    def title_page(self):
        self.add_page()
        self.set_font("DejaVu", size=24)
        self.cell(0, 60, "Bedrijfsconfigurator Rapport", 0, 1, "C")
        self.set_font("DejaVu", size=18)
        self.cell(0, 10, "Overzicht van financiële prestaties", 0, 1, "C")
        self.ln(10)
        self.set_font("DejaVu", size=14)
        self.cell(0, 10, f"Datum: {datetime.now().strftime('%d-%m-%Y')}", 0, 1, "C")
        self.ln(20)

    def chapter_title(self, num, title):
        self.set_font("DejaVu", size=16)
        self.cell(0, 10, f"Hoofdstuk {num}: {title}", 0, 1, 'L')
        self.ln(5)

    def chapter_subtitle(self, subtitle):
        self.set_font("DejaVu", size=12)
        self.cell(0, 10, subtitle, 0, 1, 'L')
        self.ln(3)

    def add_table(self, data, col_widths=None):
        self.set_font("DejaVu", size=9)
        row_height = self.font_size + 2
        
        # Check if the table fits on the page, if not, add a new page
        if self.get_y() > 250 - (row_height * len(data)):
            self.add_page()
        
        for i, row in data.iterrows():
            for column in data.columns:
                text = f"€{data[column].iloc[i]:,.2f}" if isinstance(data[column].iloc[i], (int, float)) else str(data[column].iloc[i])
                self.cell(col_widths[data.columns.get_loc(column)], row_height, text, border=1, align='C')
            self.ln(row_height)

    def add_table_with_headers(self, data):
        col_widths = [self.w / (len(data.columns) + 1)] * len(data.columns)
        row_height = self.font_size + 2

        # Check if the table fits on the page, if not, add a new page
        if self.get_y() > 250 - (row_height * (len(data) + 1)):
            self.add_page()

        # Add headers
        for column in data.columns:
            self.cell(col_widths[data.columns.get_loc(column)], row_height, str(column), border=1, align='C')
        self.ln(row_height)

        # Add the table body
        self.add_table(data, col_widths)

    def add_content_table(self, chapters):
        self.set_font("DejaVu", size=14)
        self.cell(0, 10, "Inhoudsopgave", 0, 1, 'L')
        self.ln(5)
        self.set_font("DejaVu", size=12)
        for chapter_num, chapter_title in chapters.items():
            self.cell(0, 10, f"Hoofdstuk {chapter_num}: {chapter_title}", 0, 1, 'L')
        self.ln(10)

    def add_specifications(self, specifications):
        self.set_font("DejaVu", size=10)
        for month, spec in specifications.items():
            self.chapter_subtitle(f"Specificaties voor {month}")
            for key, value in spec.items():
                self.cell(0, 8, f"{key}: €{value:,.2f}", 0, 1, 'L')
            self.ln(5)

    def add_summary_graph(self, df):
        fig, ax = plt.subplots(figsize=(8, 4))
        df_sorted = df.sort_values('month_parsed')
        df_sorted.plot(x='maand', y=['omzet', 'brutomarge', 'resultaat'], ax=ax)
        ax.set_title('Omzet, Marge en Resultaat Overzicht')
        ax.set_ylabel('Bedrag (€)')
        ax.set_xlabel('Maand')
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
            fig.savefig(temp_file.name, bbox_inches='tight')
            self.image(temp_file.name, x=10, y=None, w=180)
        plt.close(fig)

# Function to generate the report
def genereer_rapport(df, specifications):
    # Add a new column for parsed month to sort correctly
    df['month_parsed'] = df['maand'].apply(parse_dutch_month)

    pdf = PDF()
    # Ensure the DejaVu font path is correct. Adjust the path if necessary.
    try:
        pdf.add_font("DejaVu", "", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", uni=True)
    except:
        # If the above path doesn't work, you might need to adjust it based on your environment.
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)

    # Title page
    pdf.title_page()

    # Table of contents on page 2
    pdf.add_page()
    chapters = {
        1: "Inleiding",
        2: "Financiële Overzichten",
        3: "Specificaties",
        4: "Samenvattende Grafiek"
    }
    pdf.add_content_table(chapters)
    
    # Chapter 1: Introduction
    pdf.add_page()
    pdf.chapter_title(1, "Inleiding")
    inleiding_tekst = (
        "Dit rapport geeft een uitgebreid overzicht van de financiële prestaties van het bedrijf "
        "over de afgelopen maanden. Het doel van dit rapport is om inzicht te geven in de omzet, "
        "marges, kosten en resultaten per maand, evenals de vaste maandelijkse kosten. "
        "De informatie in dit rapport is bedoeld om beslissingsondersteuning te bieden en om een "
        "duidelijk beeld te geven van de financiële gezondheid van het bedrijf.\n\n"
        "In het hoofdstuk 'Financiële Overzichten' worden de maandelijkse financiële gegevens "
        "gepresenteerd, inclusief omzet, marges en kosten. Daarnaast worden er grafieken getoond "
        "om trends en verhoudingen te visualiseren.\n\n"
        "We hopen dat dit rapport u helpt bij het nemen van geïnformeerde beslissingen en het verbeteren "
        "van de financiële prestaties van het bedrijf."
    )
    pdf.multi_cell(0, 10, inleiding_tekst)
    pdf.ln(10)

    # Chapter 2: Financial Overview
    pdf.chapter_title(2, "Financiële Overzichten")
    sorted_months = df.sort_values('month_parsed')['maand'].unique()
    for month in sorted_months:
        month_data = df[df['maand'] == month]

        pdf.chapter_subtitle(f"Maand: {month}")
        financial_overview = pd.DataFrame({
            "Categorie": ["Totale Omzet", "Totale Marge", "Totale Kosten", "Resultaat"],
            "Bedrag": [
                f"€{month_data['omzet'].values[0]:,.2f}",
                f"€{month_data['brutomarge'].values[0]:,.2f}",
                f"€{month_data['kostprijs'].values[0] * -1:,.2f}",
                f"€{month_data['resultaat'].values[0]:,.2f}"
            ]
        })
        pdf.add_table_with_headers(financial_overview)
        pdf.ln(10)

        # Additional tables for each month
        omzet_marges = pd.DataFrame({
            "Categorie": ["Omzet Laadpalen", "Marge Laadpalen", "Omzet Zonnepanelen", "Marge Zonnepanelen"],
            "Bedrag": [
                f"€{month_data['omzet_laadpalen'].values[0]:,.2f}",
                f"€{month_data['brutomarge_laadpalen'].values[0]:,.2f}",
                f"€{month_data['omzet_zonnepanelen'].values[0]:,.2f}",
                f"€{month_data['brutomarge_zonnepanelen'].values[0]:,.2f}"
            ]
        })
        pdf.chapter_subtitle("Omzet en Marges")
        pdf.add_table_with_headers(omzet_marges)
        pdf.ln(10)

        kostenoverzicht = pd.DataFrame({
            "Categorie": ["Personeelskosten", "IT Kosten", "Solar Kosten", "Contributie Installatiebedrijf", "Afschrijving Vervoersmiddelen"],
            "Bedrag": [
                f"€{month_data['personeelskosten'].values[0]:,.2f}",
                f"€{-month_data['it_kosten'].values[0]:,.2f}",
                f"€{-month_data['solar_kosten'].values[0]:,.2f}",
                f"€{-month_data['contributie_kosten'].values[0]:,.2f}",
                f"€{-month_data['afschrijving_kosten'].values[0]:,.2f}"
            ]
        })
        pdf.chapter_subtitle("Kostenoverzicht")
        pdf.add_table_with_headers(kostenoverzicht)
        pdf.ln(10)

    # Chapter 4: Summary Graph
    pdf.chapter_title(4, "Samenvattende Grafiek")
    pdf.add_summary_graph(df)

    return pdf.output(dest='S').encode('latin1')

# Button to generate and download the report
if st.button("Genereer Rapport"):
    pdf_content = genereer_rapport(df_sorted, st.session_state.specificaties)
    st.download_button(
        label="Download PDF",
        data=pdf_content,
        file_name="rapport.pdf",
        mime="application/pdf"
    )
