"""PDF profesional de los resultados de InventIA."""
from io import BytesIO
from datetime import datetime
import re
import textwrap
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, PageBreak



def _imagen_grafico(grafico):
    """Convierte Plotly a PNG sin depender de Kaleido ni Chrome."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=130)
    traces = list(grafico.data)
    if not traces:
        plt.close(fig)
        raise ValueError("Gráfico sin series")
    tipo = traces[0].type
    if tipo == "pie":
        tr = traces[0]
        labels = [str(v) for v in tr.labels]
        values = [float(v) for v in tr.values]
        hole = float(getattr(tr, "hole", 0) or 0)
        ax.pie(values, labels=labels, autopct=lambda pct: f"{pct:.2f}%" if pct >= 0.5 else "",
               startangle=90, wedgeprops={"width": 1-hole} if hole else None)
        ax.axis("equal")
    else:
        for tr in traces:
            x = list(tr.x) if tr.x is not None else []
            y = list(tr.y) if tr.y is not None else []
            if not x or not y: continue
            name = str(tr.name) if tr.name else None
            if tipo == "bar":
                if str(getattr(tr,"orientation", "v")) == "h":
                    ax.barh([str(v) for v in y], [float(v) for v in x], label=name)
                else:
                    ax.bar([str(v) for v in x], [float(v) for v in y], label=name)
            elif tipo == "scatter":
                modo = str(getattr(tr, "mode", "") or "")
                if "lines" in modo:
                    ax.plot(x, y, marker="o" if "markers" in modo else None, label=name)
                    if getattr(tr, "fill", None) in ("tozeroy", "tonexty"):
                        ax.fill_between(x, y, alpha=.2)
                else:
                    ax.scatter(x, y, label=name)
            else:
                plt.close(fig)
                raise ValueError("Tipo de gráfico no compatible: " + tipo)
        if len(traces) > 1:
            ax.legend()
        ax.grid(axis="y", alpha=.2)
    titulo = getattr(grafico.layout.title, "text", None)
    if titulo: ax.set_title(str(titulo))
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return buffer



def _graficos(grafico):
    if grafico is None:
        return []
    return list(grafico) if isinstance(grafico, (list, tuple)) else [grafico]


def _grafico_pdf(grafico):
    """Dibuja la misma orientación y los mismos datos que Plotly, sin recortar nombres."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from reportlab.platypus import Image
    series = list(grafico.data)
    if not series:
        raise ValueError("Gráfico vacío")
    tipo = str(series[0].type)
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=145)
    meta = getattr(grafico.layout, "meta", None)
    metrica = meta.get("inventia_metrica", "") if isinstance(meta, dict) else ""
    unidad = "Valor del inventario" if metrica == "valor_stock" else "Unidades" if metrica in ("stock_actual", "cantidad", "cantidad_movimiento") else str(metrica).replace("_", " ").title()
    if tipo == "pie":
        tr = series[0]
        labels = [str(x) for x in tr.labels]
        vals = [float(v) for v in tr.values]
        ax.pie(vals, startangle=90, autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
               pctdistance=.78, wedgeprops={"width": .55} if float(getattr(tr, "hole", 0) or 0) else None)
        ax.legend([f"{x} — {v:,.2f} ({v/sum(vals):.1%})" for x, v in zip(labels, vals)],
                  loc="center left", bbox_to_anchor=(1.01,.5), fontsize=8, frameon=False)
        ax.set_aspect("equal")
    elif tipo == "bar":
        for tr in series:
            horizontal = str(getattr(tr, "orientation", "v") or "v") == "h"
            xs, ys = list(tr.x), list(tr.y)
            if horizontal:
                nombres = [str(y) for y in ys]
                valores = [float(x) for x in xs]
                ax.barh(nombres, valores)
                ax.set_xlabel(unidad)
                ax.invert_yaxis()
                for i,v in enumerate(valores): ax.text(v, i, f" {v:,.2f}", va="center", fontsize=8)
            else:
                nombres = [str(x) for x in xs]
                valores = [float(y) for y in ys]
                ax.bar(range(len(valores)), valores)
                ax.set_xticks(range(len(nombres)), labels=["\n".join(textwrap.wrap(x, 20)) for x in nombres], fontsize=8)
                ax.set_ylabel(unidad)
                for i,v in enumerate(valores): ax.text(i, v, f"{v:,.2f}", ha="center", va="bottom", fontsize=8)
                ax.set_ylim(0, max(valores + [1]) * 1.18)
    elif tipo == "scatter":
        for tr in series:
            if "lines" in str(getattr(tr, "mode", "") or ""):
                ax.plot(list(tr.x), list(tr.y), marker="o")
            else:
                ax.scatter(list(tr.x), list(tr.y))
        ax.set_ylabel(unidad)
    else:
        plt.close(fig)
        raise ValueError(f"Gráfico no soportado: {tipo}")
    fig.tight_layout(pad=1.8)
    if tipo == "pie": fig.subplots_adjust(right=.55)
    buf=BytesIO()
    fig.savefig(buf, format="png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image as PILImage
    im=PILImage.open(buf)
    w,h=im.size
    buf.seek(0)
    ancho=490
    alto=ancho*h/w
    if alto > 340:
        ancho *= 340/alto
        alto = 340
    return Image(buf, width=ancho, height=alto)


def generar_pdf(pregunta, respuesta, tablas, grafico):
    salida = BytesIO()
    doc = SimpleDocTemplate(salida, pagesize=(595.28,841.89), rightMargin=45,
                            leftMargin=45, topMargin=48, bottomMargin=42)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="IA_Titulo",fontName="Helvetica-Bold",fontSize=21,
                              leading=26,textColor=colors.HexColor("#173858"),spaceAfter=8))
    styles.add(ParagraphStyle(name="IA_Sub",fontName="Helvetica-Bold",fontSize=12,
                              leading=16,textColor=colors.HexColor("#087F8C"),spaceBefore=14,spaceAfter=8))
    styles.add(ParagraphStyle(name="IA_Texto",fontName="Helvetica",fontSize=9,
                              leading=14,spaceAfter=8))
    story = [Paragraph("InventIA  |  Informe de inventarios",styles["IA_Titulo"]),
             Paragraph("Generado: "+datetime.now().strftime("%d/%m/%Y %H:%M"),styles["IA_Texto"]),
             Spacer(1,12),Paragraph("Consulta analizada",styles["IA_Sub"]),
             Paragraph(escape(pregunta),styles["IA_Texto"]),
             Paragraph("Análisis",styles["IA_Sub"])]
    for linea in respuesta.splitlines():
        linea = linea.strip().replace("**","")
        if linea:
            story.append(Paragraph(escape(linea),styles["IA_Texto"]))
    # Si el informe compara almacenes, incluir conclusión y gráfico sin depender de Kaleido.
    cantidades = re.findall(r"(?:\*\*)?(Almac[eé]n [^\n:*]+)(?:\*\*)?\s*:\s*([\d,]+)", respuesta, re.I)
    if len(cantidades) >= 2:
        pares = [(nombre.strip(), int(cifra.replace(",", ""))) for nombre,cifra in cantidades]
        total = sum(c for _,c in pares)
        if total > 0:
            mayor = max(pares, key=lambda item:item[1])
            porcentaje = mayor[1]/total*100
            story.append(Paragraph("Conclusión del análisis",styles["IA_Sub"]))
            story.append(Paragraph(escape(
                f"{mayor[0]} concentra el {porcentaje:.2f}% de las unidades mostradas. "
                "Conviene revisar si esta distribución responde a las necesidades de cada almacén."
            ),styles["IA_Texto"]))
            dibujo = Drawing(495, 165)
            barras = HorizontalBarChart()
            barras.x, barras.y = 155, 25
            barras.width, barras.height = 310, 115
            barras.data = [[c for _,c in pares]]
            barras.categoryAxis.categoryNames = [nombre for nombre,_ in pares]
            barras.categoryAxis.labels.fontSize = 8
            barras.valueAxis.valueMin = 0
            barras.bars[0].fillColor = colors.HexColor("#087F8C")
            dibujo.add(barras)
            story.append(Paragraph("Distribución de existencias",styles["IA_Sub"]))
            story.append(dibujo)
    for titulo, df in tablas:
        story.append(Paragraph(escape(titulo),styles["IA_Sub"]))
        if df.empty:
            story.append(Paragraph("No se encontraron registros.",styles["IA_Texto"]))
            continue
        vista = df.head(20).copy()
        # Ajustar el ancho de la tabla y los textos extensos
        if len(vista.columns)>5:
            vista = vista.iloc[:,:5]
        columnas = [str(c).replace("_"," ").title() for c in vista.columns]
        datos = [[Paragraph(escape(c),styles["IA_Texto"]) for c in columnas]]
        for fila in vista.itertuples(index=False,name=None):
            datos.append([Paragraph(escape(str(v))[:75],styles["IA_Texto"]) for v in fila])
        ancho = (595.28-90)/len(columnas)
        tabla = Table(datos,colWidths=[ancho]*len(columnas),repeatRows=1,hAlign="LEFT")
        tabla.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EAF2F6")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F7F9FC")]),
            ("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("TOPPADDING",(0,0),(-1,-1),6),
            ("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor("#C9DCE5"))
        ]))
        story.append(tabla)
    if grafico is not None:
        story.append(Paragraph("Gráfico del análisis",styles["IA_Sub"]))
        try:
            # Plotly -> PNG. Kaleido puede requerir Chrome instalado.
            
            for figura in _graficos(grafico):
                story.append(_grafico_pdf(figura))
        except Exception:
            story.append(Paragraph(
                "El gráfico no pudo incrustarse. Para exportar imágenes de Plotly, "
                "instala Kaleido y sus dependencias de navegador.",styles["IA_Texto"]))
    story.append(Spacer(1,16))
    story.append(Paragraph("Fuente: base de datos Supabase consultada por InventIA. "
                           "Los resultados reflejan los datos disponibles al realizar la consulta.",
                           styles["IA_Texto"]))
    doc.build(story)
    salida.seek(0)
    return salida.getvalue()


def generar_pdf_conversacion(historial):
    """Informe paginado; conserva las figuras originales y todas las filas de las tablas."""
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import CondPageBreak, KeepTogether
    salida = BytesIO()
    ancho_pagina, alto_pagina = A4
    doc = SimpleDocTemplate(salida, pagesize=A4, rightMargin=40,
                            leftMargin=40, topMargin=52, bottomMargin=49)
    estilos = getSampleStyleSheet()
    estilos.add(ParagraphStyle(name="IATituloNuevo", fontName="Helvetica-Bold", fontSize=19,
                               leading=24, textColor=colors.HexColor("#153858"), spaceAfter=9))
    estilos.add(ParagraphStyle(name="IAPreguntaNueva", fontName="Helvetica-Bold", fontSize=11,
                               leading=15, textColor=colors.HexColor("#087F8C"), spaceBefore=13, spaceAfter=8))
    estilos.add(ParagraphStyle(name="IATextoNuevo", fontName="Helvetica", fontSize=9,
                               leading=13, textColor=colors.HexColor("#23374B"), spaceAfter=5))
    estilos.add(ParagraphStyle(name="IACeldaNueva", fontName="Helvetica", fontSize=7.5,
                               leading=10, textColor=colors.HexColor("#23374B")))
    estilos.add(ParagraphStyle(name="IACabeceraNueva", parent=estilos["IACeldaNueva"],
                               fontName="Helvetica-Bold", textColor=colors.HexColor("#153858")))
    story = [Paragraph("InventIA | Informe de conversación", estilos["IATituloNuevo"]),
             Paragraph("Generado: " + datetime.now().strftime("%d/%m/%Y %H:%M"), estilos["IATextoNuevo"]),
             Paragraph(f"Consultas incluidas: {len(historial)}", estilos["IATextoNuevo"]), Spacer(1, 10)]
    disponible = ancho_pagina - doc.leftMargin - doc.rightMargin
    for i, item in enumerate(historial, 1):
        story.append(CondPageBreak(115))
        story.append(Paragraph(f"Consulta {i}: " + escape(str(item.get("pregunta", ""))), estilos["IAPreguntaNueva"]))
        for linea in str(item.get("respuesta", "")).splitlines():
            linea = linea.strip().replace("**", "")
            if linea:
                story.append(Paragraph(escape(linea), estilos["IATextoNuevo"]))
        for titulo, df in item.get("tablas", []):
            story.append(CondPageBreak(90))
            story.append(Paragraph(escape(str(titulo)), estilos["IAPreguntaNueva"]))
            if df is None or df.empty:
                story.append(Paragraph("No se encontraron registros.", estilos["IATextoNuevo"]))
                continue
            # No cortar columnas ni filas; ReportLab divide tablas largas entre páginas.
            columnas = list(df.columns)
            pesos = [max(1.0, min(2.4, max(len(str(c)), *(len(str(v)) for v in df[c].head(20))) / 15)) for c in columnas]
            total_pesos = sum(pesos)
            anchos = [disponible * w / total_pesos for w in pesos]
            filas = [[Paragraph(escape(str(c).replace("_", " ").title()), estilos["IACabeceraNueva"]) for c in columnas]]
            for valores in df.itertuples(index=False, name=None):
                filas.append([Paragraph(escape("" if v is None else str(v)), estilos["IACeldaNueva"]) for v in valores])
            tabla = Table(filas, colWidths=anchos, repeatRows=1, hAlign="LEFT", splitByRow=1)
            tabla.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EAF2F6")),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F7F9FC")]),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("TOPPADDING", (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                ("LEFTPADDING", (0,0), (-1,-1), 5),
                ("RIGHTPADDING", (0,0), (-1,-1), 5),
                ("LINEBELOW", (0,0), (-1,0), .5, colors.HexColor("#B6C9D8")),
            ]))
            story.append(tabla)
        for figura in _graficos(item.get("grafico")):
            story.append(CondPageBreak(185))
            imagen = _grafico_pdf(figura)
            if imagen.drawWidth > disponible:
                escala = disponible / imagen.drawWidth
                imagen.drawWidth *= escala
                imagen.drawHeight *= escala
            story.append(imagen)
            story.append(Spacer(1, 9))
        story.append(Spacer(1, 12))
    story.append(Paragraph("Fuente: inventario consultado por InventIA.", estilos["IATextoNuevo"]))
    def pie(canvas, documento):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(40, 28, "InventIA | Informe de inventario")
        canvas.drawRightString(ancho_pagina-40, 28, f"Página {documento.page}")
        canvas.restoreState()
    doc.build(story, onFirstPage=pie, onLaterPages=pie)
    return salida.getvalue()
