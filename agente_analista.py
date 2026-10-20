import os
import json
from dotenv import load_dotenv
from groq import Groq
from datos import consultar_productos

load_dotenv()

def analizar_inventario(pregunta):
    productos = consultar_productos()
    productos = [
    {
        "codigo": p["codigo"],
        "nombre": p["nombre"],
        "precio_compra": p["precio_compra"],
        "precio_venta": p["precio_venta"],
        "estado": p["estado"]
    }
    for p in productos
]
    cliente = Groq(api_key=os.getenv("GROQ_API_KEY"))

    respuesta = cliente.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un analista experto en inventarios de productos "
                    "de limpieza. Responde siempre en español. "
                    "Utiliza exclusivamente los datos proporcionados. "
                    "No inventes cantidades de stock ni movimientos. "
                    "Si faltan datos para responder, indícalo claramente. "
                    "Explica los resultados de forma clara y profesional."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Datos de productos:\n"
                    f"{json.dumps(productos, ensure_ascii=False, default=str)}\n\n"
                    f"Pregunta: {pregunta}"
                )
            }
        ],
        temperature=0.2
    )

    return respuesta.choices[0].message.content