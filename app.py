import os
import json
import re
from datetime import datetime
from decimal import Decimal

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from datos import consultar_inventario, consultar_productos, conectar_supabase
from reportes import generar_pdf
from agente_correos import enviar_informe
from motor_ia import obtener_datos, consultar

load_dotenv()
st.set_page_config(page_title="InventIA | Inteligencia de inventarios", page_icon="✦", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],[data-testid="stApp"] {font-family:Inter,sans-serif!important}
[data-testid="stAppViewContainer"] {background:radial-gradient(ellipse at 85% 0%,#153d66 0%,transparent 42%),#06182e;color:#e6f3ff}
[data-testid="stHeader"] {background:transparent}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#071d3a,#041127)!important;border-right:1px solid #17436a}
[data-testid="stSidebar"] * {color:#e8f3ff!important}
[data-testid="stSidebar"] [role="radiogroup"] label {border:1px solid #214469;border-radius:12px;padding:13px!important;margin-bottom:10px;background:#0d2b50}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:linear-gradient(95deg,#075fa8,#073c77);border-color:#10bde3}
.block-container {max-width:1380px;padding:2rem 2.5rem 5rem}
h1,h2,h3,h4 {color:#f1f8ff!important;letter-spacing:-.025em}
p,li,small,label,[data-testid="stCaptionContainer"] {color:#c5d8eb}
[data-testid="stMetric"] {background:linear-gradient(145deg,#102f54,#0b2546);padding:18px 21px;border:1px solid #255179;border-radius:16px;box-shadow:0 12px 35px #020b1d50}
[data-testid="stMetricLabel"] p {color:#a9c6e5!important;font-size:13px}
[data-testid="stMetricValue"] {color:#fff!important}
[data-testid="stChatMessage"] {background:#0b294a;border:1px solid #204d73;border-radius:17px;padding:18px}
[data-testid="stChatMessage"] p {color:#f2f8ff}
[data-testid="stChatInput"] {background:#092343;border:1px solid #1778a6;border-radius:14px}
[data-testid="stChatInput"] textarea {color:#fff!important}
[data-testid="stDataFrame"] {border:1px solid #255179;border-radius:10px}
.stButton>button {border:1px solid #236493!important;background:#10365f!important;color:white!important;border-radius:12px!important;min-height:43px}
.stButton>button:hover {background:#145387!important;border-color:#20c6e8!important}
.stButton>button[kind="primary"],[data-testid="stFormSubmitButton"] button {background:linear-gradient(90deg,#077ac6,#00a9c7)!important;border:0!important}
.stDownloadButton>button {background:#047fa9!important;color:white!important}
[data-testid="stForm"] {background:#0b294a;border:1px solid #255179;padding:20px;border-radius:16px}
input {color:#eaf7ff!important}
/* Contraste consistente con widgets nativos de Streamlit */
[data-testid="stAppViewContainer"] .stMarkdown p,
[data-testid="stAppViewContainer"] .stMarkdown li,
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
[data-testid="stAppViewContainer"] [data-testid="stChatMessage"] * {color:#eaf5ff!important}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea::placeholder {color:#eaf5ff!important;opacity:1!important}
[data-testid="stAppViewContainer"] [data-baseweb="input"] input {color:#102e4a!important}
[data-testid="stDataFrame"] {background:#fff!important}
[data-testid="stSidebar"] [role="radiogroup"] label * {color:#f4fbff!important}
[data-testid="stAppViewContainer"] .stButton button p {color:#fff!important}
[data-testid="stAppViewContainer"] .stDownloadButton button p {color:#fff!important}
.hero {background:linear-gradient(110deg,#102d58,#064e74);border:1px solid #2373a0;box-shadow:0 14px 40px #0004;color:white;padding:29px 34px;border-radius:20px;margin-bottom:23px}
.hero h2 {color:white!important;margin:0 0 10px;font-size:29px;font-weight:800}
.hero p {margin:0;color:#c8e6fa;font-size:14px}
.hero-tag {color:#48d9f4;font-size:12px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;margin-bottom:9px}
.suggest-head {color:#96c3e3;font-size:12px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;margin:17px 0 10px}
[data-testid="stAlert"] {border-radius:13px}

[data-testid="stAppViewContainer"] .stMarkdown p,
[data-testid="stAppViewContainer"] .stMarkdown li,
[data-testid="stAppViewContainer"] .stCaption,
[data-testid="stAppViewContainer"] label {color:#e8f3ff!important}
[data-testid="stChatInput"] textarea::placeholder {color:#b9d3e7!important}
[data-testid="stAppViewContainer"] [data-baseweb="input"] input {color:#e8f3ff!important}
/* CORRECCION DEFINITIVA DEL CUADRO DE PREGUNTAS */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="textarea"] > div {
  background: #ffffff !important;
  background-color: #ffffff !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:active {
  color: #101827 !important;
  -webkit-text-fill-color: #101827 !important;
  caret-color: #101827 !important;
  background: #ffffff !important;
  opacity: 1 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: #526174 !important;
  -webkit-text-fill-color: #526174 !important;
  opacity: 1 !important;
}
[data-testid="stChatInput"] button {color:#075985 !important;}
[data-testid="stForm"] [data-baseweb="input"], [data-testid="stForm"] [data-baseweb="input"] > div {background:#fff!important}
[data-testid="stForm"] input {color:#101827!important;-webkit-text-fill-color:#101827!important;background:#fff!important;caret-color:#101827!important}
[data-testid="stForm"] input::placeholder {color:#64748b!important;-webkit-text-fill-color:#64748b!important}

/* Selector de consulta: fondo blanco y texto negro, incluida la lista desplegable */
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"] {
  background:#ffffff!important; color:#101827!important;
  -webkit-text-fill-color:#101827!important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] div,
[data-testid="stSelectbox"] [data-baseweb="select"] input {
  color:#101827!important; -webkit-text-fill-color:#101827!important;
}
[data-testid="stSelectbox"] svg {fill:#101827!important;color:#101827!important}
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] [role="listbox"] li,
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] [role="option"] {background:#ffffff!important;color:#101827!important;-webkit-text-fill-color:#101827!important}
[data-baseweb="popover"] [role="option"] *,
[data-baseweb="menu"] [role="option"] * {color:#101827!important;-webkit-text-fill-color:#101827!important}
[data-baseweb="popover"] [aria-selected="true"],
[data-baseweb="menu"] [aria-selected="true"] {background:#dcecf9!important}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="menu"] [role="option"]:hover {background:#eaf3fa!important}

/* Contraste garantizado en formulario y selector de consulta */
[data-testid="stForm"] input, [data-testid="stForm"] input:focus {color:#111827!important;-webkit-text-fill-color:#111827!important;background:#fff!important}
[data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"],
[data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"] * {color:#111827!important;-webkit-text-fill-color:#111827!important;background:#fff!important}
[role="listbox"], [role="listbox"] [role="option"], [role="listbox"] [role="option"] * {color:#111827!important;-webkit-text-fill-color:#111827!important;background:#fff!important}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60, show_spinner=False)
def cargar_inventario():
    df = pd.DataFrame(consultar_inventario())
    if df.empty:
        return df
    for campo in ["stock_actual", "stock_minimo", "stock_maximo", "valor_stock",
                  "precio_compra", "precio_venta"]:
        if campo in df:
            df[campo] = pd.to_numeric(df[campo], errors="coerce").fillna(0)
    return df

@st.cache_data(ttl=60, show_spinner=False)
def cargar_movimientos():
    try:
        datos = conectar_supabase().table("movimientos_inventario").select("*").range(0,999).execute().data
        return pd.DataFrame(datos)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def cargar_todas_tablas():
    return obtener_datos()

def num(valor):
    return f"{valor:,.0f}".replace(",", ".")

def moneda(valor):
    return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def resumen(df):
    if df.empty:
        return {}
    return {
        "productos_unicos": int(df["id_producto"].nunique()),
        "registros": len(df),
        "almacenes": int(df["almacen"].nunique()),
        "existencias": float(df["stock_actual"].sum()),
        "valor_inventario": float(df["valor_stock"].sum()),
        "registros_bajo_minimo": int((df["stock_actual"] < df["stock_minimo"]).sum()),
        "registros_en_minimo": int((df["stock_actual"] == df["stock_minimo"]).sum()),
    }

def estilo_grafico(figura):
    """Fondo oscuro y textos de alto contraste para Plotly."""
    figura.update_layout(
        template="plotly_dark", paper_bgcolor="#0b294a", plot_bgcolor="#0b294a",
        font=dict(color="#eff8ff", size=13),
        margin=dict(l=12, r=20, t=35, b=25),
        legend=dict(font=dict(color="#eff8ff")),
    )
    figura.update_xaxes(gridcolor="#244d70", zerolinecolor="#244d70")
    figura.update_yaxes(gridcolor="#244d70", zerolinecolor="#244d70")
    return figura


def analizar_pregunta(pregunta, df, movimientos):
    """Cálculos reales en Python; Groq solo redacta la interpretación."""
    q = pregunta.lower()
    base = resumen(df)
    tablas = []
    grafico = None
    hallazgos = {"indicadores": base}
    if any(w in q for w in ["movimiento", "entrada", "salida", "vendid", "ventas"]):
        if movimientos.empty:
            hallazgos["advertencia"] = "No se pudieron consultar los movimientos. No es posible inferir ventas."
        else:
            tipos = next((c for c in ["tipo_movimiento", "tipo", "tipo_operacion"] if c in movimientos), None)
            if tipos:
                conteo = movimientos[tipos].astype(str).value_counts().reset_index()
                conteo.columns = ["tipo", "cantidad_registros"]
                tablas.append(("Movimientos registrados", conteo))
                hallazgos["movimientos_por_tipo"] = conteo.to_dict("records")
                grafico = px.bar(conteo, x="tipo", y="cantidad_registros",
                                 color="tipo", color_discrete_sequence=["#087F8C","#234B72"])
            else:
                hallazgos["advertencia"] = "No se reconoce la columna de tipo de movimiento."
                hallazgos["columnas_movimientos"] = list(movimientos.columns)
    elif any(w in q for w in ["mínim", "minim", "bajo stock", "agotad", "reposici", "reponer"]):
        t = df.loc[df["stock_actual"] < df["stock_minimo"],
                   ["producto","almacen","stock_actual","stock_minimo"]].copy()
        t["faltante_hasta_minimo"] = t["stock_minimo"] - t["stock_actual"]
        tablas.append(("Productos bajo el mínimo", t))
        hallazgos["bajo_minimo"] = t.head(30).to_dict("records")
    elif any(w in q for w in ["almac", "depósito", "deposito", "distribuci"]):
        t = df.groupby("almacen",as_index=False).agg(
            unidades=("stock_actual","sum"), valor=("valor_stock","sum"),
            productos=("id_producto","nunique"))
        tablas.append(("Inventario por almacén",t))
        hallazgos["por_almacen"] = t.to_dict("records")
        hallazgos["metodo"] = "Totales exactos por almacén calculados en Python con todos los registros recibidos."
        grafico = px.bar(t,x="almacen",y="unidades",color="almacen",
                         color_discrete_sequence=["#087F8C","#234B72"])
    elif any(w in q for w in ["valor", "costo", "coste", "dinero", "inversi"]):
        t = df.groupby("producto",as_index=False)["valor_stock"].sum().nlargest(10,"valor_stock")
        tablas.append(("Productos de mayor valor",t))
        hallazgos["top_valor"] = t.to_dict("records")
        grafico = px.bar(t,x="valor_stock",y="producto",orientation="h",
                         color_discrete_sequence=["#087F8C"])
    elif any(w in q for w in ["producto", "stock", "existenc", "cantidad", "mayor", "menor", "top"]):
        t = df.groupby("producto",as_index=False)["stock_actual"].sum()
        t = t.nsmallest(10,"stock_actual") if any(w in q for w in ["menor","menos","bajo"]) else t.nlargest(10,"stock_actual")
        tablas.append(("Existencias por producto",t))
        hallazgos["productos_destacados"] = t.to_dict("records")
        hallazgos["metodo"] = "Ranking calculado a partir de TODOS los registros de la vista, agrupados por producto. Los resultados son los 10 extremos globales, no una muestra arbitraria."
        grafico = px.bar(t,x="stock_actual",y="producto",orientation="h",
                         color_discrete_sequence=["#087F8C"])
    else:
        hallazgos["nota"] = "Consulta general. Solo hay datos agregados del inventario en esta respuesta."
    if grafico is not None:
        grafico = estilo_grafico(grafico)
    return hallazgos, tablas, grafico

def respuesta_ia(pregunta, hallazgos):
    cliente = Groq(api_key=os.getenv("GROQ_API_KEY"))
    mensajes = [
        {"role":"system","content":(
            "Eres InventIA, un analista profesional de inventarios de productos de limpieza. "
            "Responde en español, de manera clara, útil y breve. Usa exclusivamente los datos "
            "calculados que recibes. No inventes cifras, fechas, ventas, tendencias ni causas. "
            "Si los datos no permiten responder exactamente, dilo y explica qué falta. "
            "Distingue productos únicos de registros por almacén. No uses formato JSON en la respuesta. "
            "IMPORTANTE: los rankings que recibes ya se calcularon con el inventario completo; "
            "NO afirmes que falta el resto de productos. NO cambies los valores de las tablas. "
            "Si hay un ranking, explica directamente sus primeros resultados con sus cifras exactas.")},
        {"role":"user","content":"Pregunta: "+pregunta+"\nDatos comprobados: "+
         json.dumps(hallazgos,ensure_ascii=False,default=str)}
    ]
    r = cliente.chat.completions.create(model=os.getenv("GROQ_MODEL","openai/gpt-oss-20b"),
                                        messages=mensajes,temperature=0.1,max_tokens=650)
    return r.choices[0].message.content

with st.sidebar:
    st.markdown("## ✦ Invent**IA**")
    st.caption("PLATAFORMA DE INTELIGENCIA ARTIFICIAL")
    st.divider()
    seccion = st.radio("Navegación",["💬 Asistente inteligente","📄 Informes inteligentes"],
                       label_visibility="collapsed")
    st.divider()
    st.caption("Conectado a Supabase · Datos de solo lectura")
    if st.button("↻ Actualizar información",use_container_width=True):
        cargar_inventario.clear()
        cargar_movimientos.clear()
        cargar_todas_tablas.clear()
        st.rerun()

try:
    inventario = cargar_inventario()
    if inventario.empty:
        st.warning("No hay registros de inventario disponibles.")
        st.stop()
except Exception as exc:
    st.error(f"No se pudo consultar Supabase: {exc}")
    st.stop()

if "historial" not in st.session_state:
    st.session_state.historial = []
if "ultimo" not in st.session_state:
    st.session_state.ultimo = None

if seccion.startswith("💬"):
    st.markdown('<div class="hero"><div class="hero-tag">✦ Asistente inteligente</div><h2>Tu inventario, explicado con inteligencia.</h2>'
                '<p>Haz preguntas sobre productos, existencias, almacenes y movimientos. '
                'Recibe análisis basados en datos reales.</p></div>',unsafe_allow_html=True)
    k = resumen(inventario)
    c1,c2,c3 = st.columns(3)
    c1.metric("Productos",num(k["productos_unicos"]))
    c2.metric("Existencias registradas",num(k["existencias"]))
    c3.metric("Valor del inventario",moneda(k["valor_inventario"]))
    st.caption("Las existencias pueden incluir distintas unidades de medida. El valor se muestra sin símbolo de moneda.")
    for item in st.session_state.historial:
        with st.chat_message("user"):
            st.write(item["pregunta"])
        with st.chat_message("assistant"):
            st.markdown(item["respuesta"])
            for titulo, tabla in item["tablas"]:
                st.caption(titulo)
                st.dataframe(tabla,use_container_width=True,hide_index=True)
            if item["grafico"] is not None:
                for indice, figura in enumerate(item["grafico"] if isinstance(item["grafico"], (list, tuple)) else [item["grafico"]]):
                    st.plotly_chart(figura,use_container_width=True, key=f"grafico_{id(item)}_{indice}")
    pregunta = st.chat_input("Ej.: ¿Qué productos tienen menos existencias?")
    if not pregunta:
        pregunta = st.session_state.pop("pregunta_sugerida",None)
    if pregunta:
        with st.spinner("Interpretando tu pregunta y verificando los datos..."):
            try:
                datos_tablas, errores = cargar_todas_tablas()
                respuesta, tablas, grafico, detalles = consultar(
                    pregunta,datos_tablas,st.session_state.historial)
                item = {"pregunta":pregunta,"respuesta":respuesta,"tablas":tablas,
                        "grafico":grafico,"datos":detalles}
                st.session_state.historial.append(item)
                st.session_state.ultimo = item
                st.rerun()
            except Exception as exc:
                st.error(f"No fue posible responder: {exc}")
    st.markdown('<div class="suggest-head">✦ Consultas rápidas</div>',unsafe_allow_html=True)
    opciones = ["¿Qué almacén tiene más unidades?", "¿Cuáles son los 10 productos de mayor valor?",
                "¿Hay productos bajo el mínimo?", "¿Cuáles son los productos con menos existencias?"]
    cols = st.columns(2)
    for i, sugerencia in enumerate(opciones):
        with cols[i % 2]:
            if st.button("✧ " + sugerencia, key=f"sug_{i}",use_container_width=True):
                st.session_state["pregunta_sugerida"] = sugerencia
                st.rerun()

else:
    st.markdown('<div class="hero"><div class="hero-tag">✦ Documentos y comunicación</div><h2>Informes inteligentes</h2>'
                '<p>Convierte tus análisis en documentos profesionales y compártelos por correo.</p>'
                '</div>',unsafe_allow_html=True)
    historial = st.session_state.historial
    if not historial:
        st.info("Primero conversa con el Asistente inteligente; luego podrás exportar toda la conversación.")
    else:
        st.markdown("#### Contenido del informe")
        modo = st.radio("Incluir", ["Toda la conversación", "Una consulta específica"], horizontal=True)
        if modo == "Toda la conversación":
            seleccionados = historial
            st.caption(f"Se incluirán {len(historial)} preguntas con sus respuestas.")
        else:
            indice = st.selectbox("Elige la consulta", range(len(historial)),
                                  format_func=lambda i: f"{i+1}. {historial[i]['pregunta'][:110]}")
            seleccionados = [historial[indice]]
        try:
            from reportes import generar_pdf_conversacion
            pdf = generar_pdf_conversacion(seleccionados)
            st.download_button("⬇ Descargar informe PDF", pdf,
                               file_name="InventIA_conversacion.pdf", mime="application/pdf", type="primary")
            st.divider()
            st.markdown("#### Enviar por correo")
            with st.form("correo"):
                destinatario = st.text_input("Correo destinatario", placeholder="nombre@empresa.com")
                asunto = st.text_input("Asunto", value="Informe de inventario | InventIA")
                enviar = st.form_submit_button("Enviar informe", type="primary")
            if enviar:
                if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", destinatario.strip()):
                    st.error("Introduce una dirección de correo válida.")
                elif st.session_state.get("ultimo_envio") and (datetime.now()-st.session_state["ultimo_envio"]).total_seconds()<60:
                    st.warning("Espera un minuto antes de enviar otro informe.")
                else:
                    try:
                        enviar_informe(destinatario.strip(), asunto, pdf)
                        st.session_state["ultimo_envio"] = datetime.now()
                        st.success("El servidor de correo aceptó el informe.")
                    except Exception as exc:
                        st.error(f"No se pudo enviar el correo: {exc}")
                        if "Configura SMTP" in str(exc):
                            st.info("Completa SMTP_HOST, SMTP_PORT, SMTP_USER y SMTP_PASSWORD en el archivo .env del proyecto y reinicia Streamlit.")
        except Exception as exc:
            st.error(f"No se pudo generar el PDF: {exc}")
    st.caption("Los envíos automáticos requieren un programador externo (por ejemplo, GitHub Actions). No se ejecutan solo por dejar Streamlit abierto.")
