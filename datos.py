
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

def conectar_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("Faltan las credenciales de Supabase.")

    return create_client(url, key)

def consultar_productos():
    cliente = conectar_supabase()

    respuesta = (
        cliente.table("productos")
        .select("*")
        .execute()
    )

    return respuesta.data
def consultar_inventario():
    cliente = conectar_supabase()
    respuesta = (
        cliente.table("vista_inventario")
        .select("*")
        .execute()
    )
    return respuesta.data
def obtener_estadisticas_inventario():
    registros = consultar_inventario()

    total_unidades = sum(
        float(r["stock_actual"] or 0)
        for r in registros
    )

    valor_total = sum(
        float(r["valor_stock"] or 0)
        for r in registros
    )

    stock_bajo = [
        r for r in registros
        if float(r["stock_actual"] or 0)
        < float(r["stock_minimo"] or 0)
    ]

    almacenes = sorted({
        r["almacen"]
        for r in registros
        if r.get("almacen")
    })

    return {
        "productos_unicos": len({
            r["id_producto"] for r in registros
        }),
        "registros_inventario": len(registros),
        "total_unidades": total_unidades,
        "valor_total_inventario": round(valor_total, 2),
        "registros_stock_bajo": len(stock_bajo),
        "almacenes": almacenes
    }
