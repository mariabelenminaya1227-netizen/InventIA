"""Motor de consultas seguro: Groq interpreta; pandas calcula. Solo lectura."""
import json
import os
import re
import unicodedata
import pandas as pd
import plotly.express as px
from groq import Groq
from datetime import datetime, timedelta, timezone
from datos import conectar_supabase

TABLAS = ["vista_inventario", "productos", "movimientos_inventario",
          "categorias", "marcas", "proveedores", "almacenes"]
NUMEROS = {"stock_actual","stock_minimo","stock_maximo","valor_stock",
           "precio_compra","precio_venta","cantidad","cantidad_movimiento"}
FECHAS = {"fecha_vencimiento","fecha_actualizacion","fecha_movimiento",
          "created_at","fecha","fecha_registro"}
OPERACIONES = {"count","distinct","sum","avg","median","min","max","rows"}
VISUALES = {"texto","tabla","grafico","ambos"}

def sin_tildes(s):
    return "".join(c for c in unicodedata.normalize("NFD",str(s).lower())
                   if unicodedata.category(c)!="Mn")

def obtener_datos():
    cliente = conectar_supabase()
    datos, errores = {}, {}
    for tabla in TABLAS:
        try:
            filas=[]
            for inicio in range(0,10000,1000):
                bloque=cliente.table(tabla).select("*").range(inicio,inicio+999).execute().data or []
                filas.extend(bloque)
                if len(bloque)<1000:
                    break
            else:
                errores[tabla]="Se alcanzó el límite de lectura de 10 000 registros."
            datos[tabla]=pd.DataFrame(filas)
        except Exception as exc:
            errores[tabla]=str(exc)
    return datos,errores

def esquema(datos):
    resultado={}
    for nombre,df in datos.items():
        if df.empty:
            continue
        cats={}
        for c in df.columns:
            if c in ("almacen","categoria","estado_stock","tipo_movimiento",
                     "tipo","marca","unidad_medida"):
                cats[c]=df[c].dropna().astype(str).unique()[:35].tolist()
        resultado[nombre]={"columnas":list(df.columns),"filas":len(df),
                           "valores_disponibles":cats}
    return resultado

def planificar(pregunta, datos, historial):
    cliente=Groq(api_key=os.getenv("GROQ_API_KEY"))
    contexto=[{"pregunta":h.get("pregunta",""),"respuesta":h.get("respuesta","")[:400]}
              for h in historial[-8:]]
    instruccion="""Eres el planificador de consultas de InventIA. Devuelve SOLO JSON válido, sin markdown.
NO inventes columnas, tablas ni valores. Usa el esquema suministrado.
Formato:
{"tabla":"vista_inventario","operacion":"sum","campo":"stock_actual",
"agrupar":"almacen","filtros":[{"campo":"almacen","valor":"Almacén Principal","modo":"contiene"}],
"orden":"desc","limite":10,"visual":"texto","explicacion":"..."}
operacion: count (filas), distinct (cuenta valores únicos), sum, avg, median, min, max, rows.
agrupar: columna, lista de columnas o null. campo: columna o null. Para cruces por almacén y producto puedes usar agrupar ["almacen","producto"].
filtros: lista, cada uno modo 'contiene','igual','menor','mayor','menor_igual','mayor_igual','distinto','desde','hasta','columna_menor','columna_menor_igual','columna_mayor','columna_mayor_igual'. Para comparar dos columnas usa valor con el nombre de la otra columna. Ejemplo stock bajo mínimo: {'campo':'stock_actual','modo':'columna_menor','valor':'stock_minimo'}. Para stock cero: modo igual valor 0.
Para vencimientos próximos usa fecha_vencimiento y filtros de fecha; no inventes fecha actual.
visual: texto, tabla, grafico, ambos. SOLO usa tabla/gráfico si el usuario los pide EXPLÍCITAMENTE.
Para 'cuántos productos' usa distinct de id_producto en vista_inventario.
Para 'productos con menos stock' usa sum stock_actual agrupado por producto, asc.
Para 'qué almacén tiene más unidades' usa sum stock_actual agrupado por almacen, desc.
Para 'movimientos' usa movimientos_inventario y las columnas existentes.
No asumas que una salida de inventario equivale a una venta.
Si la pregunta es ambigua, elige la interpretación razonable y explícitala.
Para listas detalladas usa rows y columnas pertinentes.
Para rankings de valor de inventario usa sum de valor_stock agrupado por producto (o id_producto si no existe producto), orden desc y limite solicitado.
Para rankings de existencias usa sum de stock_actual agrupado por producto, orden asc/desc según la pregunta.
Para valores por almacén o categoría usa sum valor_stock agrupado por la columna solicitada.
Para alertas de reposición usa filtro columna_menor stock_actual respecto a stock_minimo; operacion rows con columnas de producto, stock_actual, stock_minimo, o count si se pide cantidad. Para stock agotado usa filtro igual a 0. Para alertas por vencimiento usa fechas reales y filtros desde/hasta. No confundas valor_stock con unidades.
Para preguntas de precios, proveedores, marcas, movimientos y fechas, elige la tabla y las columnas REALES del esquema.
No inventes columnas; para datos disponibles en una misma vista puedes agrupar por varias columnas. Si requiere unir tablas no conectadas en una vista, declara la limitación.
En filtros usa solo campos reales y valores sustentados en el esquema o la pregunta.
Si la pregunta pide riesgos, conclusiones o recomendaciones sobre una consulta anterior, conserva el contexto y no devuelvas filas de productos.
No ejecutes código ni SQL. Si no se puede responder, pon operacion 'rows' y explica limitación.
"""
    respuesta=cliente.chat.completions.create(
        model=os.getenv("GROQ_MODEL","openai/gpt-oss-20b"),temperature=0,
        max_tokens=750,
        messages=[{"role":"system","content":instruccion},
                  {"role":"user","content":json.dumps({
                      "pregunta":pregunta,"esquema":esquema(datos),
                      "historial":contexto},ensure_ascii=False,default=str)}])
    texto=(respuesta.choices[0].message.content or "").strip()
    # Acepta JSON con texto o bloques markdown alrededor; evita expresiones regulares codiciosas.
    decoder=json.JSONDecoder()
    for inicio, caracter in enumerate(texto):
        if caracter == "{":
            try:
                objeto, _ = decoder.raw_decode(texto[inicio:])
                if isinstance(objeto, dict) and objeto.get("tabla"):
                    return objeto
            except (ValueError, json.JSONDecodeError):
                continue
    raise ValueError("Groq no devolvió un plan de consulta utilizable.")

def calcular(plan,datos,pregunta):
    tabla=plan.get("tabla","vista_inventario")
    if tabla not in datos or datos[tabla].empty:
        return "No hay datos disponibles de esa tabla para responder con precisión.",[],None,{}
    df=datos[tabla].copy()
    cols=set(df.columns)
    filtros=plan.get("filtros") or []
    for filtro in filtros[:8]:
        if not isinstance(filtro, dict):
            continue
        c=filtro.get("campo")
        if c not in cols: continue
        valor=filtro.get("valor")
        modo=filtro.get("modo","contiene")
        serie=df[c]
        if modo.startswith("columna_"):
            if not isinstance(valor, str) or valor not in cols:
                return "No existe la segunda columna necesaria para comparar los datos.", [], None, {"plan":plan}
            izquierda=pd.to_numeric(serie,errors="coerce")
            derecha=pd.to_numeric(df[valor],errors="coerce")
            operadores={"columna_menor":izquierda<derecha,
                        "columna_menor_igual":izquierda<=derecha,
                        "columna_mayor":izquierda>derecha,
                        "columna_mayor_igual":izquierda>=derecha,
                         "columna_igual":izquierda.eq(derecha),
                         "columna_distinto":izquierda.ne(derecha)}
            if modo not in operadores:
                return "Comparación de columnas no admitida.", [], None, {"plan":plan}
            df=df[operadores[modo].fillna(False)]
        elif modo=="contiene":
            df=df[serie.astype(str).str.contains(str(valor),case=False,regex=False,na=False)]
        elif modo=="igual":
            df=df[serie.astype(str).str.casefold()==str(valor).casefold()]
        elif modo=="distinto":
            df=df[serie.astype(str).str.casefold()!=str(valor).casefold()]
        elif modo in ("menor","mayor","menor_igual","mayor_igual"):
            numeros=pd.to_numeric(serie,errors="coerce")
            try:
                v=float(valor)
                comparaciones={"menor":numeros<v,"mayor":numeros>v,"menor_igual":numeros<=v,"mayor_igual":numeros>=v}
                df=df[comparaciones[modo]]
            except (TypeError,ValueError):
                pass
        elif modo in ("desde","hasta"):
            fechas=pd.to_datetime(serie,errors="coerce",utc=True)
            v=pd.to_datetime(valor,errors="coerce",utc=True)
            if not pd.isna(v):
                df=df[fechas>=v] if modo=="desde" else df[fechas<=v]
    op=plan.get("operacion","rows")
    if op not in OPERACIONES: op="rows"
    campo=plan.get("campo")
    agrupar=plan.get("agrupar")
    if isinstance(agrupar, list):
        if not agrupar or len(agrupar)>3 or any(not isinstance(c,str) or c not in cols for c in agrupar):
            return "No existen todas las columnas necesarias para agrupar.", [], None, {"plan":plan}
    elif agrupar is not None and not isinstance(agrupar,str):
        return "La agrupación solicitada no es válida.", [], None, {"plan":plan}
    if campo not in cols: campo=None
    if not isinstance(agrupar,list) and agrupar not in cols:
        if agrupar is not None:
            return "No existe la columna de agrupación solicitada.", [], None, {"plan": plan}
        agrupar=None
    if op in ("sum","avg","median","min","max") and campo:
        df[campo]=pd.to_numeric(df[campo],errors="coerce")
    # No convertir un campo inválido en un conteo engañoso.
    if op in ("sum", "avg", "median", "min", "max", "distinct") and campo is None:
        return "No existe la columna necesaria para calcular esa consulta.", [], None, {"plan": plan}
    if agrupar:
        grupo=df.groupby(agrupar,dropna=False)
        if op=="count": resultado=grupo.size().reset_index(name="cantidad")
        elif op=="distinct" and campo: resultado=grupo[campo].nunique().reset_index(name="cantidad")
        elif op in ("sum","avg","median","min","max") and campo:
            resultado=getattr(grupo[campo],{"avg":"mean"}.get(op,op))().reset_index(name=campo)
        else:
            resultado=grupo.size().reset_index(name="cantidad")
        medida=resultado.columns[-1]
        resultado=resultado.sort_values(medida,ascending=plan.get("orden")=="asc")
    elif op=="count":
        resultado=pd.DataFrame({"resultado":[len(df)]})
    elif op=="distinct" and campo:
        resultado=pd.DataFrame({"resultado":[df[campo].nunique()]})
    elif op in ("sum","avg","median","min","max") and campo:
        serie=df[campo]
        valor=getattr(serie,{"avg":"mean"}.get(op,op))()
        resultado=pd.DataFrame({"resultado":[valor]})
    elif op=="rows":
        campos=plan.get("columnas") or []
        campos=[c for c in campos if c in cols]
        if not campos: campos=list(df.columns)[:8]
        resultado=df[campos].copy()
        if campo and campo in resultado and plan.get("orden") in ("asc","desc"):
            resultado=resultado.sort_values(campo,ascending=plan["orden"]=="asc")
    else:
        return "No hay información suficiente para calcular esa consulta con los campos disponibles.",[],None,{}
    # Mostrar nombres legibles al agrupar por ID de producto, sin alterar los cálculos.
    if (agrupar == "id_producto" or isinstance(agrupar,list) and "id_producto" in agrupar) and "id_producto" in resultado.columns:
        nombres = None
        if "producto" in df.columns:
            nombres = df[["id_producto", "producto"]].dropna(subset=["id_producto", "producto"]).drop_duplicates("id_producto").set_index("id_producto")["producto"]
        elif "productos" in datos and not datos["productos"].empty:
            catalogo = datos["productos"]
            columna_nombre = next((c for c in ("nombre", "nombre_producto", "producto", "descripcion") if c in catalogo.columns), None)
            if columna_nombre and "id_producto" in catalogo.columns:
                nombres = catalogo[["id_producto", columna_nombre]].dropna().drop_duplicates("id_producto").set_index("id_producto")[columna_nombre]
        if nombres is not None:
            def nombre_producto(identificador):
                try:
                    valor = nombres.get(identificador)
                    if valor is not None and str(valor).strip():
                        return str(valor)
                except (TypeError, ValueError):
                    pass
                return str(identificador)
            resultado["id_producto"] = resultado["id_producto"].map(nombre_producto)
            resultado = resultado.rename(columns={"id_producto": "producto"})
    try:
        limite=max(1,min(int(plan.get("limite") or 10),100))
    except (ValueError,TypeError):
        limite=10
    pregunta_normal=sin_tildes(pregunta)
    if any(t in pregunta_normal for t in ("conclusion", "compar", "cual tiene mas", "distribu", "recomend", "analiza")):
        limite=max(limite,20)
    if op == "rows" and not plan.get("limite"):
        limite = len(resultado)
    resultado=resultado.head(limite)
    visual=plan.get("visual","texto")
    # La intención explícita del usuario tiene prioridad sobre el planificador.
    normal=sin_tildes(pregunta)
    pide_grafico=bool(re.search(r"\b(grafico|graficos|grafica|graficas|graficar|diagrama|visualiza|plot|barras|circular|pastel|torta|donut|lineas|dispersion)\b",normal))
    pide_tabla=bool(re.search(r"\b(tabla|cuadro|tabular|hoja de calculo)\b",normal))
    if pide_grafico and pide_tabla: visual="ambos"
    elif pide_grafico: visual="grafico"
    elif pide_tabla: visual="tabla"
    else: visual="texto"
    grafico=None
    if visual in ("grafico", "ambos") and len(resultado.columns) >= 2 and not resultado.empty:
        x, y = resultado.columns[:2]
        # Para agrupaciones múltiples, mostrar una etiqueta combinada sin perder el dato tabular.
        if len(resultado.columns)>2 and pd.api.types.is_numeric_dtype(resultado.iloc[:,-1]):
            nombres=resultado.iloc[:,:-1].fillna("sin dato").astype(str).agg(" / ".join,axis=1)
            graf_df=pd.DataFrame({"grupo":nombres,"valor":resultado.iloc[:,-1].values})
            x,y="grupo","valor"
        else:
            graf_df=resultado
        if pd.api.types.is_numeric_dtype(graf_df[y]):
            if any(t in normal for t in ("circular", "pastel", "torta", "pie", "donut", "anillo")):
                grafico = px.pie(graf_df, names=x, values=y,
                                 hole=.48 if any(t in normal for t in ("donut", "anillo")) else 0)
            elif any(t in normal for t in ("horizontal", "barras acostadas")):
                grafico = px.bar(graf_df, x=y, y=x, orientation="h")
            elif any(t in normal for t in ("lineas", "linea", "tendencia", "evolucion")):
                grafico = px.line(graf_df, x=x, y=y, markers=True)
            elif any(t in normal for t in ("area", "área")):
                grafico = px.area(graf_df, x=x, y=y)
            elif any(t in normal for t in ("dispersion", "dispersión", "scatter")):
                grafico = px.scatter(graf_df, x=x, y=y)
            else:
                grafico = px.bar(graf_df, x=x, y=y)
            # La métrica viaja con el gráfico para que el PDF no etiquete valores como unidades.
            grafico.update_layout(meta={"inventia_metrica": str(campo or y), "inventia_categoria": str(x)})
            unidad = "valor del inventario" if (campo or y) == "valor_stock" else "unidades" if (campo or y) in ("stock_actual", "cantidad", "cantidad_movimiento") else str(campo or y).replace("_", " ")
            # Etiquetas legibles: nunca redondear 99.96% a 100%.
            if grafico.data and grafico.data[0].type == "pie":
                grafico.update_traces(textinfo="percent", texttemplate="%{percent:.2%}",
                                      hovertemplate=f"%{{label}}: %{{value:,.2f}} {unidad} (%{{percent:.2%}})<extra></extra>")
            elif grafico.data and grafico.data[0].type == "bar":
                grafico.update_traces(texttemplate="%{x:,.0f}" if grafico.data[0].orientation == "h" else "%{y:,.0f}",
                                      textposition="outside", cliponaxis=False)
                if grafico.data[0].orientation == "h":
                    grafico.update_layout(xaxis_title=unidad.title(), yaxis_title="Almacén" if x == "almacen" else str(x))
                    grafico.update_xaxes(range=[0, float(pd.to_numeric(graf_df[y], errors="coerce").max()) * 1.22 or 1])
                else:
                    grafico.update_layout(xaxis_title=str(x).replace("_", " ").title(), yaxis_title=unidad.title())
                    grafico.update_yaxes(range=[0, float(pd.to_numeric(graf_df[y], errors="coerce").max()) * 1.22 or 1])
            grafico.update_layout(template="plotly_dark", paper_bgcolor="#0c2949",
                                  plot_bgcolor="#0c2949", font_color="#f4f8ff", height=400)
    # Porcentajes calculados sobre el conjunto completo, nunca sobre la muestra visible.
    # Se añaden DESPUÉS de construir la figura: el gráfico mantiene la métrica original.
    pide_porcentaje = bool(re.search(r"\b(porcentaje|porcentajes|porcentual|participacion|proporcion|%|peso relativo)\b", normal))
    if pide_porcentaje and agrupar and len(resultado.columns) == 2 and pd.api.types.is_numeric_dtype(resultado.iloc[:, 1]):
        # Para que la suma sea coherente con la tabla, mostrar todos los grupos en este caso.
        # El cálculo de la proporción se hace antes del redondeo.
        if len(resultado) < len(df.groupby(agrupar, dropna=False)):
            grupo_completo = df.groupby(agrupar, dropna=False)
            if op == "count":
                resultado = grupo_completo.size().reset_index(name="cantidad")
            elif op == "distinct" and campo:
                resultado = grupo_completo[campo].nunique().reset_index(name="cantidad")
            elif op in ("sum", "avg", "median", "min", "max") and campo:
                resultado = getattr(grupo_completo[campo], {"avg": "mean"}.get(op, op))().reset_index(name=campo)
            resultado = resultado.sort_values(resultado.columns[-1], ascending=plan.get("orden") == "asc")
            # Recrear la figura a partir del mismo conjunto que aparece en la tabla.
            if grafico is not None:
                medida = resultado.columns[-1]
                categoria = resultado.columns[0]
                if grafico.data[0].type == "pie":
                    grafico = px.pie(resultado, names=categoria, values=medida)
                    grafico.update_traces(texttemplate="%{percent:.2%}", textinfo="percent",
                                          hovertemplate="%{label}: %{value:,.2f} (%{percent:.2%})<extra></extra>")
                else:
                    grafico = px.bar(resultado, x=categoria, y=medida)
                    grafico.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
                grafico.update_layout(meta={"inventia_metrica": str(campo or medida)},
                                      template="plotly_dark", paper_bgcolor="#0c2949",
                                      plot_bgcolor="#0c2949", font_color="#f4f8ff",
                                      legend_font_color="#f4f8ff", height=430)
        metrica = resultado.columns[1]
        total = pd.to_numeric(resultado[metrica], errors="coerce").sum()
        resultado["porcentaje (%)"] = (pd.to_numeric(resultado[metrica], errors="coerce") / total * 100).round(2) if total else 0.0
        if grafico is not None:
            grafico.update_layout(font_color="#f4f8ff", legend_font_color="#f4f8ff")
        # Una pregunta porcentual requiere tabla de porcentajes, incluso si también hay gráfico.
        if visual == "grafico":
            visual = "ambos"
    def mostrar(v):
        if pd.isna(v): return "sin dato"
        if isinstance(v,(float,int)) and not isinstance(v,bool):
            return f"{v:,.2f}".rstrip("0").rstrip(".") if isinstance(v,float) else f"{v:,}"
        return str(v)
    if pide_porcentaje and agrupar and "porcentaje (%)" in resultado.columns and not resultado.empty:
        primero = resultado.iloc[0]
        texto = (f"**{primero.iloc[0]}** representa el **{float(primero['porcentaje (%)']):.2f}%** "
                 f"del total calculado. La tabla muestra las cantidades y los porcentajes "
                 f"de todos los grupos (total: {float(total):,.2f}).")
    elif resultado.empty:
        texto="No se encontraron registros que cumplan las condiciones de tu consulta."
    elif len(resultado)==1 and list(resultado.columns)==["resultado"]:
        texto=f"**Resultado: {mostrar(resultado.iloc[0,0])}**."
    elif len(resultado.columns)==2:
        pares=[f"**{mostrar(fila[0])}**: {mostrar(fila[1])}"
               for fila in resultado.itertuples(index=False,name=None)]
        texto="\n".join("- "+x for x in pares)
        # Conclusiones basadas en los resultados calculados, sin inventar cifras.
        desea_analisis=any(t in normal for t in
            ("conclusion", "compar", "cual tiene mas", "cual tiene menos", "distribu", "recomend", "analiza", "diferencia"))
        if desea_analisis and pd.api.types.is_numeric_dtype(resultado.iloc[:,1]):
            valores=pd.to_numeric(resultado.iloc[:,1],errors="coerce").fillna(0)
            orden=valores.sort_values(ascending=False)
            if len(orden)>=2:
                imax, imin=orden.index[0], orden.index[-1]
                mayor, menor=float(valores.loc[imax]),float(valores.loc[imin])
                nombre_mayor=str(resultado.loc[imax].iloc[0])
                nombre_menor=str(resultado.loc[imin].iloc[0])
                total=float(valores.sum())
                diferencia=mayor-menor
                texto+=(f"\n\n**Comparación:** {nombre_mayor} registra la mayor cantidad "
                        f"({mostrar(mayor)}), frente a {nombre_menor} "
                        f"({mostrar(menor)}). La diferencia es de {mostrar(diferencia)} unidades.")
                if total>0:
                    texto+=f" {nombre_mayor} representa el {mayor/total*100:.2f}% del total mostrado."
                if any(t in normal for t in ("conclusion","recomend","analiza","distribu")):
                    if total>0 and mayor/total>=0.90:
                        texto+=("\n\n**Conclusión:** las existencias están muy concentradas en "
                                f"{nombre_mayor}. **Recomendación:** revisar si esta distribución "
                                "responde a la demanda y a la función de cada almacén; si ambos "
                                "atienden pedidos, evaluar una redistribución.")
                    else:
                        texto+=("\n\n**Conclusión:** hay diferencias de existencias entre los "
                                "grupos analizados. **Recomendación:** contrastar la distribución "
                                "con la demanda y la capacidad operativa antes de realizar cambios.")
            elif len(orden)==1:
                texto+="\n\nSolo hay un grupo en el resultado; no es posible comparar dos grupos."
        elif len(resultado)>10:
            texto+=f"\n\n*Se presentan {len(resultado)} resultados.*"
    else:
        texto=f"Encontré **{len(resultado)} registros** relacionados con tu consulta."
        if visual=="texto":
            for fila in resultado.head(12).to_dict("records"):
                texto+="\n- "+", ".join(f"{k}: {mostrar(v)}" for k,v in fila.items())
    if tabla=="movimientos_inventario" and "venta" in normal:
        texto+="\n\n*Nota: una salida de inventario no necesariamente equivale a una venta.*"
    # Una misma consulta puede pedir circular Y barras: devolver ambas figuras.
    if grafico is not None and re.search(r"\b(circular|pastel|torta|donut|anillo)\b", normal) and re.search(r"\b(barras|barra)\b", normal):
        categorias = resultado.columns[0]
        metrica = resultado.columns[-1]
        if len(resultado.columns) == 2 and pd.api.types.is_numeric_dtype(resultado[metrica]):
            barras = px.bar(resultado, x=categorias, y=metrica)
            barras.update_layout(meta={"inventia_metrica": str(campo or metrica)},
                                 xaxis_title=str(categorias).replace("_", " ").title(),
                                 yaxis_title=unidad.title(), template="plotly_dark",
                                 paper_bgcolor="#0c2949", plot_bgcolor="#0c2949",
                                 font_color="#f4f8ff", height=400)
            barras.update_traces(texttemplate="%{y:,.0f}", textposition="outside", cliponaxis=False)
            grafico = [grafico, barras]
    tablas=[("Resultados",resultado)] if visual in ("tabla","ambos") else []
    return texto,tablas,grafico,{"plan":plan,"filas_filtradas":len(df)}




def _cliente_ia():
    clave = os.getenv("GROQ_API_KEY")
    if not clave:
        raise ValueError("Falta GROQ_API_KEY en la configuración de InventIA.")
    return Groq(api_key=clave)


def _json_ia(texto):
    coincidencia = re.search(r"\{.*\}", texto, re.S)
    if not coincidencia:
        raise ValueError("La IA no devolvió un plan válido. Intenta de nuevo.")
    return json.loads(coincidencia.group())


def _historial_contexto(historial):
    mensajes = []
    for item in historial[-10:]:
        mensajes.append({"pregunta": str(item.get("pregunta", ""))[:600],
                         "respuesta": str(item.get("respuesta", ""))[:2400]})
    return mensajes


def _decidir_intencion(pregunta, historial):
    """El modelo decide si debe calcular nuevos datos o continuar razonando."""
    instrucciones = (
        "Eres el enrutador conversacional de InventIA, especialista en inventarios. "
        "Responde SOLO JSON: {\"accion\":\"analizar\" o \"consultar\",\"pregunta_datos\":\"...\"}. "
        "Usa 'analizar' para preguntas de seguimiento que pidan explicaciones, riesgos, "
        "prioridades, decisiones, propuestas, escenarios hipotéticos o recomendaciones "
        "sobre información YA presente en el historial, aunque no mencionen nombres. "
        "Usa 'consultar' si se solicitan nuevas cifras, productos, filtros, listados, "
        "comparaciones con datos no calculados o un gráfico/tabla que requiere nueva consulta. "
        "Si se pide una visualización de resultados previos, usa 'consultar' y reescribe "
        "pregunta_datos incluyendo la métrica y agrupación del historial. "
        "Para consultar, reescribe pregunta_datos como consulta autónoma que conserve "
        "referencias del contexto (por ejemplo, 'ese almacén'). "
        "No inventes datos. Para analizar, pregunta_datos puede quedar vacía."
    )
    respuesta = _cliente_ia().chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), temperature=0,
        max_tokens=900,
        messages=[{"role": "system", "content": instrucciones},
                  {"role": "user", "content": json.dumps({
                      "historial": _historial_contexto(historial), "pregunta": pregunta
                  }, ensure_ascii=False)}])
    decision = _json_ia(respuesta.choices[0].message.content or "")
    if decision.get("accion") not in ("analizar", "consultar"):
        raise ValueError("No se pudo determinar la intención de la consulta.")
    return decision


def _redactar(pregunta, historial, evidencia, solo_contexto=False):
    """Redacción fundamentada: los números provienen de pandas o del historial."""
    sistema = (
        "Eres InventIA, un analista de inventarios que conversa en español de forma "
        "natural, clara y profesional. Contesta directamente la pregunta actual y "
        "mantén el hilo de la conversación. Puedes razonar sobre riesgos, acciones, "
        "causas posibles, prioridades y escenarios, distinguiendo hipótesis de hechos. "
        "NO inventes números, productos, almacenes, fechas, movimientos ni métricas. "
        "Solo afirma hechos sustentados por EVIDENCIA o HISTORIAL. Si falta un dato "
        "indispensable, explica qué dato falta. No muestres identificadores internos "
        "ni enumeraciones de registros salvo petición expresa. No afirmes que un "
        "movimiento de salida es una venta. No inventes moneda. "
        "Para preguntas de seguimiento, evita repetir listas de existencias si no aportan. "
        "Si piden un número concreto de acciones, respétalo y priorízalas. "
        "Nunca digas que has ejecutado acciones operativas. "
        "No digas 'Según los datos de Supabase' como muletilla. "
        "Si se solicita un gráfico o tabla, describe brevemente el resultado, NUNCA des instrucciones para crearlo ni digas que el usuario debe seleccionar opciones en otro panel. "
        "No incluyas una tabla Markdown ni un gráfico ASCII; la interfaz los muestra "
        "separadamente cuando el usuario los solicita."
    )
    carga = {"pregunta": pregunta, "historial": _historial_contexto(historial),
             "evidencia": evidencia[:12000], "modo": "seguimiento" if solo_contexto else "consulta"}
    mensajes = [{"role": "system", "content": sistema},
                {"role": "user", "content": json.dumps(carga, ensure_ascii=False, default=str)}]
    # Los modelos con razonamiento pueden agotar el presupuesto antes de producir content.
    # Reintentar con mayor presupuesto, sin perder el historial ni inventar una respuesta.
    for presupuesto in (2200, 4000):
        respuesta = _cliente_ia().chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            temperature=0.2, max_tokens=presupuesto, messages=mensajes)
        texto = (respuesta.choices[0].message.content or "").strip()
        if texto:
            return texto
    return ("No pude completar la respuesta de análisis en este momento. "
            "Conservo el contexto de la conversación; vuelve a solicitar el análisis.")


def _grafico_personalizado(plan, pregunta, tablas, grafico):
    """El tipo de gráfico solicitado se determina por el lenguaje del usuario."""
    if grafico is None or not tablas:
        return grafico
    return grafico


def _visual_desde_contexto(pregunta, datos, historial):
    """Gráficos de seguimiento: usa planes comprobables, sin depender de texto de Groq."""
    normal = sin_tildes(pregunta)
    quiere_grafico = bool(re.search(r"\b(grafico|graficos|grafica|graficas|barras|circular|pastel|torta|donut|anillo|horizontal|lineas|linea|area|dispersion)\b", normal))
    if not quiere_grafico:
        return None
    referencia = bool(re.search(r"\b(ahora|muestralo|muestrala|cambialo|conviertelo|esa|ese|eso|mismo|misma|distribucion)\b", normal))
    # Priorizar el plan real de una visualización o consulta anterior.
    plan = None
    if referencia:
        for item in reversed(historial):
            anterior = (item.get("datos") or {}).get("plan")
            if isinstance(anterior, dict) and anterior.get("agrupar"):
                plan = dict(anterior)
                break
    if plan is None:
        return None
    plan["visual"] = "grafico"
    resumen, tablas, grafico, detalles = calcular(plan, datos, pregunta)
    if grafico is None:
        return None
    etiquetas = {"circular": "circular", "pastel": "circular", "donut": "de dona",
                 "horizontal": "de barras horizontales", "lineas": "de líneas",
                 "area": "de áreas", "dispersion": "de dispersión"}
    tipo = next((etiqueta for palabra, etiqueta in etiquetas.items() if palabra in normal), "solicitado")
    detalles["plan"] = plan
    return (f"Aquí tienes el gráfico {tipo} con los datos de inventario de la consulta anterior.",
            tablas, grafico, detalles)


def _grafico_obligatorio(pregunta, datos, historial):
    """Ejecuta solicitudes visuales con pandas/Plotly, nunca con texto generado por IA.

    Para cambios de tipo conserva el último plan numérico; para una referencia
    a distribución por almacén recupera los datos actuales de Supabase.
    """
    n = sin_tildes(pregunta)
    if not re.search(r"\b(grafico|graficos|grafica|graficas|barras|circular|pastel|torta|donut|anillo|horizontal|lineas|linea|area|dispersion|scatter)\b", n):
        return None
    vista = datos.get("vista_inventario")
    if vista is None or vista.empty:
        return ("No hay datos de inventario disponibles para generar el gráfico.", [], None, {"tipo": "sin_datos"})
    # El usuario puede pedir un nuevo tema: no reutilizar otro plan por accidente.
    explicito_almacenes = bool(re.search(r"\balmacen(?:es)?\b", n))
    es_seguimiento = bool(re.search(r"\b(ahora|muestralo|muestrala|cambialo|conviertelo|ese|esa|eso|mismo|misma|distribucion)\b", n))
    plan = None
    if es_seguimiento and not explicito_almacenes:
        for h in reversed(historial):
            candidato = (h.get("datos") or {}).get("plan")
            if isinstance(candidato, dict) and candidato.get("agrupar") and candidato.get("tabla") in datos:
                plan = dict(candidato)
                break
    contexto = " ".join(str(h.get("pregunta", "")) for h in historial[-8:])
    if plan is None and (explicito_almacenes or (es_seguimiento and "almacen" in sin_tildes(contexto))):
        if {"almacen", "stock_actual"}.issubset(vista.columns):
            plan = {"tabla": "vista_inventario", "operacion": "sum", "campo": "stock_actual",
                    "agrupar": "almacen", "filtros": [], "orden": "desc", "limite": 30}
    if plan is None:
        return None  # Otras peticiones pasan al planificador general.
    plan["visual"] = "grafico"
    texto, tablas, figura, detalles = calcular(plan, datos, pregunta)
    if figura is None:
        return ("No pude generar un gráfico con los datos disponibles para esa consulta.", [], None, detalles)
    tipo = ("circular" if any(w in n for w in ("circular", "pastel", "torta")) else
            "de dona" if any(w in n for w in ("donut", "anillo")) else
            "de barras horizontales" if "horizontal" in n else
            "de líneas" if any(w in n for w in ("linea", "lineas")) else
            "de áreas" if "area" in n else "de dispersión" if "dispersion" in n else "de barras")
    detalles["plan"] = plan
    if re.search(r"\b(porcentaje|porcentajes|porcentual|participacion|proporcion)\b", n):
        return texto, tablas, figura, detalles
    return ("Aquí tienes el gráfico " + tipo + " generado con los datos del inventario.",
            tablas, figura, detalles)


def _seguimiento_formato(pregunta, datos, historial):
    """Reutiliza la consulta anterior cuando solo cambia su presentación.

    Nunca depende de que Groq vuelva a interpretar «esa información».
    """
    if not historial:
        return None
    n = sin_tildes(pregunta)
    pide_tabla = bool(re.search(r"\b(tabla|cuadro|tabular|formato tabular)\b", n))
    pide_grafico = bool(re.search(r"\b(grafico|grafica|barras|circular|pastel|donut|lineas)\b", n))
    if not (pide_tabla or pide_grafico):
        return None
    # No interceptar una consulta nueva con tema propio.
    referencia = bool(re.search(
        r"\b(esa|ese|eso|esta|este|esto|anterior|previo|misma|mismo|"
        r"muestralo|muestrala|damelo|ponlo|conviertelo|ahora|informacion|resultados)\b", n))
    # Una petición con métricas nuevas NO es un cambio de formato.
    if re.search(r"\b(igual|iguales|exactamente|limite|debajo|inferior|agotad|vencim|proxim|ordenad)\b", n):
        return None
    if ("stock" in n and "minim" in n and
            re.search(r"\b(productos?|articulos?)\b", n)):
        return None
    # "Dame una tabla de productos agotados" es una consulta nueva, no seguimiento.
    if re.search(r"\b(tabla|grafico|grafica)\s+(de|por|sobre)\s+(productos|almacenes|categorias|movimientos|proveedores|marcas)\b", n) and not re.search(r"\b(esa|ese|anterior|misma|mismo)\b",n):
        return None
    tema_nuevo = bool(re.search(r"\b(de|por|sobre)\s+(otro|otra|nuevos|nuevas)\b", n))
    # Petición que solo menciona tipos de gráficos: usa la última consulta, no inventa un tema.
    solo_formato = bool(re.fullmatch(r"[\w\s,¿?¡!]+", n)) and not re.search(r"\b(stock|precio|vencimiento|almacen|categoria|proveedor|movimiento|producto)\b", n)
    if (not referencia and not solo_formato) or tema_nuevo:
        return None
    for item in reversed(historial):
        plan = (item.get("datos") or {}).get("plan")
        if not isinstance(plan, dict) or plan.get("tabla") not in datos:
            continue
        nuevo = dict(plan)
        nuevo["visual"] = "ambos" if pide_tabla and pide_grafico else "tabla" if pide_tabla else "grafico"
        # El motor calcula sobre los datos reales, no sobre una tabla inventada por IA.
        texto, tablas, grafico, detalles = calcular(nuevo, datos, pregunta)
        if pide_tabla and not tablas:
            continue
        if pide_grafico and grafico is None:
            continue
        detalles["plan"] = nuevo
        if pide_tabla and pide_grafico:
            mensaje = "Aquí tienes la tabla y el gráfico de la consulta anterior, calculados con el inventario."
        elif pide_tabla:
            mensaje = "Aquí tienes los resultados de la consulta anterior en una tabla."
        else:
            mensaje = "Aquí tienes el gráfico de la consulta anterior."
        return mensaje, tablas, grafico, detalles
    return None


def _consulta_stock_en_limite(pregunta, datos):
    """Compara columnas reales antes de cualquier reutilización del historial."""
    n = sin_tildes(pregunta)
    if not ("stock" in n and "minim" in n and
            re.search(r"\b(igual|iguales|exactamente|justo|limite|nivel minimo)\b", n)):
        return None
    vista = datos.get("vista_inventario")
    if vista is None or vista.empty:
        return ("No hay datos de inventario para comprobar los mínimos.", [], None, {"tipo": "sin_datos"})
    requeridas = ("producto", "almacen", "stock_actual", "stock_minimo")
    faltan = [c for c in requeridas if c not in vista.columns]
    if faltan:
        return ("No puedo comprobar el stock mínimo: faltan " + ", ".join(faltan) + ".", [], None,
                {"tipo": "columnas_faltantes"})
    actual = pd.to_numeric(vista["stock_actual"], errors="coerce")
    minimo = pd.to_numeric(vista["stock_minimo"], errors="coerce")
    mascara = actual.notna() & minimo.notna() & actual.eq(minimo)
    detalle = vista.loc[mascara, list(requeridas)].copy()
    if "orden" in n or "alfabet" in n:
        detalle = detalle.sort_values(["producto", "almacen"], kind="stable")
    cantidad = len(detalle)
    if cantidad == 0:
        return ("No hay productos exactamente en su stock mínimo.", [], None,
                {"tipo": "stock_en_limite", "filas_filtradas": 0})
    cero = int((pd.to_numeric(detalle["stock_minimo"], errors="coerce") == 0).sum())
    texto = (f"Encontré {cantidad} registros con stock actual igual al mínimo. "
             f"De ellos, {cero} tienen mínimo configurado en cero; "
             "no deben interpretarse automáticamente como existencias suficientes.")
    return (texto, [("Productos exactamente en el stock mínimo", detalle)], None,
            {"tipo": "stock_en_limite", "filas_filtradas": cantidad})


def _consulta_detalle_stock(pregunta, datos):
    """Atiende solicitudes explícitas de detalle por producto sin reutilizar agregados."""
    n = sin_tildes(pregunta)
    if not (re.search(r"\b(productos?|articulos?)\b", n)
            and "stock" in n and "minim" in n
            and any(t in n for t in ("actual", "almacen", "detalle", "lista", "tabla", "cada producto"))):
        return None
    # Una pregunta sobre productos bajo el mínimo se resuelve por la regla específica.
    if any(t in n for t in ("debajo", "por debajo", "inferior", "menor que", "bajo el", "reposicion urgente")):
        return None
    vista = datos.get("vista_inventario")
    if vista is None or vista.empty:
        return ("No hay registros de inventario disponibles para mostrar el detalle por producto.", [], None,
                {"tipo": "sin_datos"})
    obligatorias = ("stock_actual", "stock_minimo", "almacen")
    faltantes = [c for c in obligatorias if c not in vista.columns]
    if "producto" not in vista.columns and "id_producto" not in vista.columns:
        faltantes.append("producto o id_producto")
    if faltantes:
        return ("No puedo elaborar esa tabla: faltan las columnas " + ", ".join(faltantes)
                + " en vista_inventario.", [], None, {"tipo": "columnas_faltantes"})
    identificador = "producto" if "producto" in vista.columns else "id_producto"
    columnas = [identificador, "almacen", "stock_actual", "stock_minimo"]
    detalle = vista[columnas].copy()
    if identificador == "id_producto":
        catalogo = datos.get("productos")
        if catalogo is not None and not catalogo.empty and "id_producto" in catalogo.columns:
            nombre = next((c for c in ("nombre", "nombre_producto", "producto") if c in catalogo.columns), None)
            if nombre:
                detalle = detalle.merge(catalogo[["id_producto", nombre]].drop_duplicates("id_producto"),
                                        on="id_producto", how="left")
                detalle[nombre] = detalle[nombre].fillna(detalle["id_producto"].astype(str))
                detalle = detalle.drop(columns="id_producto").rename(columns={nombre: "producto"})
                detalle = detalle[["producto", "almacen", "stock_actual", "stock_minimo"]]
    total = len(detalle)
    # Mostrar todos los registros disponibles; el PDF puede paginar la tabla.
    return (f"Detalle de {total} registros por producto y almacén, con stock actual y mínimo.",
            [("Stock actual y mínimo por producto", detalle)], None,
            {"tipo": "detalle_stock", "filas_filtradas": total})


def _consulta_comprobable(pregunta, datos):
    """Resuelve preguntas críticas con columnas reales, sin inventar un plan IA."""
    n = sin_tildes(pregunta)
    vista = datos.get("vista_inventario")
    if vista is None or vista.empty:
        return None
    cols = set(vista.columns)
    visual = "ambos" if "tabla" in n and "grafico" in n else "tabla" if "tabla" in n else "grafico" if "grafico" in n else "texto"
    if "venc" in n and any(w in n for w in ("proxim", "pronto", "caducar", "vencer")):
        if "fecha_vencimiento" not in cols:
            return ("No puedo identificar próximos vencimientos: falta la columna fecha_vencimiento en la vista de inventario.", [], None, {"tipo":"columna_faltante"})
        df = vista.copy()
        fechas = pd.to_datetime(df["fecha_vencimiento"], errors="coerce", utc=True)
        hoy = pd.Timestamp.now(tz="UTC").normalize()
        dias = 30
        coincidencia = re.search(r"(\d+)\s*dias", n)
        if coincidencia: dias = min(3650, int(coincidencia.group(1)))
        df = df.loc[(fechas >= hoy) & (fechas <= hoy + pd.Timedelta(days=dias))].copy()
        if df.empty:
            return (f"No hay productos con fecha de vencimiento registrada en los próximos {dias} días.", [], None, {"tipo":"vencimientos"})
        df["fecha_vencimiento"] = fechas.loc[df.index].dt.strftime("%Y-%m-%d")
        campos = [c for c in ("producto", "almacen", "fecha_vencimiento", "stock_actual") if c in df.columns]
        df = df.sort_values("fecha_vencimiento")[campos].head(100)
        return (f"Encontré {len(df)} registros con vencimiento en los próximos {dias} días.", [("Próximos vencimientos", df)], None, {"tipo":"vencimientos"})
    if "stock" in n and "minim" in n and any(w in n for w in ("debajo", "bajo", "menor", "inferior", "reponer", "reposicion")):
        if not {"stock_actual", "stock_minimo"}.issubset(cols):
            return ("No puedo comparar existencias y mínimos: faltan stock_actual o stock_minimo.", [], None, {"tipo":"columna_faltante"})
        plan={"tabla":"vista_inventario", "operacion":"rows", "campo":"stock_actual", "columnas":[c for c in ("producto", "almacen", "stock_actual", "stock_minimo") if c in cols], "filtros":[{"campo":"stock_actual", "modo":"columna_menor", "valor":"stock_minimo"}], "orden":"asc", "limite":100, "visual":"tabla"}
        texto, tablas, grafico, detalle = calcular(plan, datos, pregunta + " en tabla")
        if tablas and tablas[0][1].empty:
            texto = "No hay registros por debajo del stock mínimo entre los datos consultados."
        else:
            texto = f"Encontré {len(tablas[0][1])} registros por debajo del stock mínimo. La comparación se hizo por registro y almacén."
        return texto, tablas, grafico, detalle
    if "producto" in n and "caro" in n:
        campo = next((c for c in ("precio_venta", "precio_compra") if c in cols), None)
        if campo is None:
            return ("No puedo determinar los productos más caros: no hay precio_venta ni precio_compra en la vista.", [], None, {"tipo":"columna_faltante"})
        grupo = "id_producto" if "id_producto" in cols else "producto" if "producto" in cols else None
        if grupo is None:
            return ("Falta una columna para identificar los productos.", [], None, {"tipo":"columna_faltante"})
        limite = int(re.search(r"\b(\d+)\s+productos", n).group(1)) if re.search(r"\b(\d+)\s+productos", n) else 10
        plan={"tabla":"vista_inventario", "operacion":"max", "campo":campo, "agrupar":grupo, "orden":"desc", "limite":limite, "visual":visual}
        texto, tablas, grafico, detalle = calcular(plan, datos, pregunta)
        return f"Ranking por {campo.replace('_', ' ')} (precio unitario; no son unidades de stock):\n" + texto, tablas, grafico, detalle
    return None

def consultar(pregunta, datos, historial):
    n = sin_tildes(pregunta)
    if (re.search(r"\b(porcentaje|porcentajes|porcentual|participacion|proporcion)\b", n)
            and re.search(r"\balmacen(?:es)?\b", n)
            and re.search(r"\b(stock|existencias|unidades|inventario|total)\b", n)):
        vista = datos.get("vista_inventario")
        if vista is not None and {"almacen", "stock_actual"}.issubset(vista.columns):
            plan = {"tabla": "vista_inventario", "operacion": "sum", "campo": "stock_actual",
                    "agrupar": "almacen", "orden": "desc", "limite": 10000, "filtros": [], "visual": "ambos"}
            return calcular(plan, datos, pregunta)
    en_limite = _consulta_stock_en_limite(pregunta, datos)
    if en_limite is not None:
        return en_limite
    detalle_stock = _consulta_detalle_stock(pregunta, datos)
    if detalle_stock is not None:
        return detalle_stock
    comprobable = _consulta_comprobable(pregunta, datos)
    if comprobable is not None:
        return comprobable
    formato_anterior = _seguimiento_formato(pregunta, datos, historial)
    if formato_anterior is not None:
        return formato_anterior
    grafico_directo = _grafico_obligatorio(pregunta, datos, historial)
    if grafico_directo is not None:
        return grafico_directo
    visual_contexto = _visual_desde_contexto(pregunta, datos, historial)
    if visual_contexto is not None:
        return visual_contexto
    normal = sin_tildes(pregunta)
    solicita_grafico = bool(re.search(r"\b(grafico|graficos|grafica|graficas|barras|circular|pastel|torta|donut|lineas|dispersion)\b", normal))
    # Transformar el gráfico anterior es una operación de presentación, NO una nueva consulta IA.
    if solicita_grafico and historial and re.search(r"\b(ahora|muestralo|muestrala|cambialo|conviertelo|mismo|misma|ese|esa)\b", normal):
        for anterior in reversed(historial):
            figura_previa = anterior.get("grafico")
            plan_previo = (anterior.get("datos") or {}).get("plan")
            if isinstance(plan_previo, dict) and figura_previa is not None:
                plan = dict(plan_previo)
                resumen, tablas, grafico, detalles = calcular(plan, datos, pregunta)
                if grafico is not None:
                    return ("Aquí tienes los mismos datos representados en el formato solicitado. "
                            "Puedes seguir pidiendo cambios de visualización.", tablas, grafico, detalles)
        # Si la consulta anterior era la distribución de almacenes, recuperar esa misma métrica.
        contexto = " ".join(str(h.get("pregunta", "")) for h in historial[-6:])
        vista = datos.get("vista_inventario")
        if ("almacen" in sin_tildes(contexto) and vista is not None
                and {"almacen", "stock_actual"}.issubset(vista.columns)):
            plan = {"tabla":"vista_inventario", "operacion":"sum", "campo":"stock_actual",
                    "agrupar":"almacen", "filtros":[], "orden":"desc", "limite":30, "visual":"grafico"}
            resumen, tablas, grafico, detalles = calcular(plan, datos, pregunta)
            if grafico is not None:
                return ("Aquí tienes la distribución de existencias por almacén en el gráfico solicitado.",
                        tablas, grafico, detalles)
    # Evitar que una respuesta JSON defectuosa de Groq rompa toda la conversación.
    try:
        decision = _decidir_intencion(pregunta, historial)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        es_analisis = bool(historial and re.search(
            r"\b(por que|riesgo|problema|consecuencia|recomiend|medida|accion|prioridad|conclusion|solucion|mejorar|esa situacion|esa distribucion)\b",
            sin_tildes(pregunta)))
        decision = {"accion": "analizar" if es_analisis else "consultar",
                    "pregunta_datos": pregunta}

    # Una petición explícita de gráfico/tabla nunca se resuelve solo con texto.
    normal = sin_tildes(pregunta)
    pide_visual = bool(re.search(r"\b(grafico|graficos|grafica|graficas|graficar|diagrama|visualiza|barras|circular|pastel|torta|donut|lineas|dispersion|tabla|cuadro)\b", normal))
    if pide_visual:
        decision["accion"] = "consultar"
    if decision["accion"] == "analizar":
        if not historial:
            return ("Necesito primero una consulta de inventario para analizar esa situación.",
                    [], None, {"tipo": "sin_contexto"})
        respuesta = _redactar(pregunta, historial,
                             "Analiza solo los hechos y resultados del historial; no hay nuevos cálculos.",
                             solo_contexto=True)
        return respuesta, [], None, {"tipo": "seguimiento"}

    pregunta_datos = str(decision.get("pregunta_datos") or pregunta)
    # Para una nueva representación de un gráfico anterior, reutilizar exactamente
    # sus datos calculados, sin arriesgarse a que el modelo cambie la consulta.
    cambia_formato = bool(re.search(r"\b(ahora|muestralo|muestrala|mismo|misma|lo|ese|esa)\b", normal))
    if pide_visual and cambia_formato and historial:
        for anterior in reversed(historial):
            figura_anterior = anterior.get("grafico")
            plan_anterior = (anterior.get("datos") or {}).get("plan")
            if figura_anterior is not None and isinstance(plan_anterior, dict):
                plan = dict(plan_anterior)
                resumen, tablas, grafico, detalles = calcular(plan, datos, pregunta)
                if grafico is not None:
                    evidencia = "Resultados calculados de la misma consulta anterior: " + resumen
                    respuesta = _redactar(pregunta, historial, evidencia)
                    return respuesta, tablas, grafico, detalles
                break
    # Para "ahora muéstralo en ..." reutilizar el tema previo al planificar.
    if pide_visual and historial and re.search(r"\b(ahora|muestralo|muestrala|el mismo|lo mismo|ese|esa)\b", normal):
        preguntas_previas = [str(h.get("pregunta", "")) for h in historial[-6:]]
        if preguntas_previas:
            pregunta_datos += ". Contexto de la visualización anterior: " + " | ".join(preguntas_previas[-3:])
    # Peticiones visuales sobre resultados anteriores: reutiliza un plan calculado,
    # incluso cuando el resultado previo era texto (sin figura).
    if pide_visual and historial and re.search(r"\b(esa|ese|eso|lo|muestralo|ahora|mismo|misma)\b", normal):
        for anterior in reversed(historial):
            previo=(anterior.get("datos") or {}).get("plan")
            if isinstance(previo, dict) and previo.get("agrupar"):
                resumen, tablas, grafico, detalles = calcular(dict(previo), datos, pregunta)
                if grafico is not None:
                    return ("Aquí tienes los resultados anteriores en el gráfico solicitado.", tablas, grafico, detalles)
        contexto=" ".join(str(h.get("pregunta", "")) for h in historial[-8:])
        vista=datos.get("vista_inventario")
        if ("almacen" in sin_tildes(contexto) and vista is not None
                and {"almacen", "stock_actual"}.issubset(vista.columns)):
            plan={"tabla":"vista_inventario", "operacion":"sum", "campo":"stock_actual",
                  "agrupar":"almacen", "filtros":[], "orden":"desc", "limite":30,
                  "visual":"grafico"}
            resumen, tablas, grafico, detalles=calcular(plan, datos, pregunta)
            if grafico is not None:
                return ("Aquí tienes las existencias por almacén en el gráfico solicitado.", tablas, grafico, detalles)
    # Consulta muy frecuente y bien definida: evitar que un plan ambiguo
    # convierta un gráfico de almacenes en un listado de productos.
    consulta_almacenes = bool(re.search(r"\balmacen(?:es)?\b", sin_tildes(pregunta_datos)))
    consulta_existencias = bool(re.search(r"\b(stock|existencias|unidades|inventario|distribucion)\b", sin_tildes(pregunta_datos)))
    vista = datos.get("vista_inventario")
    consulta_detallada = bool(re.search(r"\b(productos?|articulos?)\b", normal) or "stock_minimo" in normal or "stock minimo" in normal)
    if pide_visual and not consulta_detallada and consulta_almacenes and consulta_existencias and vista is not None and {"almacen", "stock_actual"}.issubset(vista.columns):
        plan = {"tabla":"vista_inventario", "operacion":"sum", "campo":"stock_actual", "agrupar":"almacen", "filtros":[], "orden":"desc", "limite":30, "visual":"grafico"}
    else:
        try:
            plan = planificar(pregunta_datos, datos, historial)
        except (ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            # No presentar un error de JSON como si la pregunta del usuario fuera incorrecta.
            # Una consulta no resuelta no debe convertirse en un listado arbitrario.
            return ("No pude interpretar la consulta con suficiente precisión. "
                    "Conservé la conversación; prueba indicando el dato o agrupación que quieres consultar.",
                    [], None, {"tipo":"plan_no_disponible"})
    # La interfaz visual depende de la petición actual, no de la reformulación.
    resumen, tablas, grafico, detalles = calcular(plan, datos, pregunta)
    if pide_visual and grafico is None and re.search(r"\b(grafico|graficos|grafica|graficas|barras|circular|pastel|donut|lineas|dispersion)\b", normal):
        # Nunca decir que se generó un gráfico si no existe un objeto Plotly.
        if not resumen.startswith("No hay"):
            resumen += "\n\nNo pude construir el gráfico con los campos devueltos; necesito una categoría y una cantidad numérica."
    # 'calcular' retorna solo una muestra para texto; los cálculos son deterministas.
    if resumen.startswith("No hay datos disponibles") or resumen.startswith("No hay información suficiente"):
        return resumen, [], None, detalles
    evidencia = ("Consulta de datos interpretada: " + pregunta_datos + "\n"
                 "Resultado calculado con pandas: " + resumen + "\n"
                 "Advertencia: los listados pueden estar limitados por el plan. "
                 "No extrapoles de una muestra al inventario completo.")
    if pide_visual and grafico is None and re.search(r"\b(grafico|graficos|grafica|graficas|barras|circular|pastel|donut|horizontal)\b", normal):
        return ("No pude generar un gráfico válido con esos datos. Indica qué variable deseas comparar.",
                [], None, detalles)
    if pide_visual and grafico is not None:
        respuesta = "Aquí tienes el gráfico generado con los datos consultados de tu inventario."
    else:
        respuesta = _redactar(pregunta, historial, evidencia)
    if pide_visual and grafico is not None:
        respuesta += "\n\n**Gráfico generado a partir de los datos del inventario.**"
    return respuesta, tablas, grafico, detalles
