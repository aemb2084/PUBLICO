# Código de predicción corregido

THRESHOLD = 0.3

test_comment = [
    # CALIDAD - preguntas nuevas
    "¿De qué manera influyen los errores de calibración en la calidad final del producto?",
    "¿Cuál es el impacto de utilizar herramientas desgastadas sobre los productos defectuosos?",
    "¿Cómo pueden los sistemas automáticos de inspección reducir las piezas rechazadas?",

    # VELOCIDAD - preguntas nuevas
    "¿Por qué razón una máquina podría estar operando por debajo de su capacidad nominal?",
    "¿Cuál es la diferencia entre el tiempo estándar y el tiempo real de ciclo?",
    "¿De qué forma los cambios de formato afectan la tasa de producción?",

    # RENDIMIENTO - preguntas nuevas
    "¿Qué tipos de paradas no planificadas impactan más en la disponibilidad del equipo?",
    "¿Cómo se calcula el tiempo neto de operación en una jornada productiva?",
    "¿Cuáles son las principales causas de tiempos muertos en una línea de producción?",

    # EFICIENCIA OEE - preguntas nuevas
    "¿Qué valor de OEE se considera aceptable para una planta manufacturera?",
    "¿Cómo se puede mejorar el indicador global de efectividad del equipo?",

    # TODAS (multi-label) - preguntas nuevas
    "¿Qué estrategias permiten optimizar simultáneamente calidad, velocidad y disponibilidad?",
    "¿Cómo impacta la capacitación del personal en todos los indicadores del OEE?",
    "¿Qué papel juega el mantenimiento autónomo en la mejora integral del OEE?",
]

# Realizar predicciones
data_module.predict_texts = test_comment
preds = trainer.predict(model, datamodule=data_module, ckpt_path="best")
all_preds = torch.cat(preds).numpy()

# Crear DataFrame con resultados
df = pd.DataFrame(all_preds, columns=LABEL_COLUMNS, index=test_comment)

# Agregar columnas con predicciones binarias (aplicando threshold)
for col in LABEL_COLUMNS:
    df[f'{col}_pred'] = (df[col] >= THRESHOLD).astype(int)

# Guardar resultados en CSV
output_file = f"{LOGS_DIR}/predicciones_test.csv"
df.to_csv(output_file, encoding='utf-8-sig')
print(f"\n✅ Resultados guardados en: {output_file}")

# Mostrar resumen
print("\n=== RESUMEN DE PREDICCIONES ===")
print(df)

# SOLUCIÓN: Crear función de estilo correcta
def highlight_predictions(row):
    """Resalta en rojo las probabilidades mayores al threshold"""
    styles = []
    for col in LABEL_COLUMNS:
        if row[col] >= THRESHOLD:
            styles.append('background-color: lightcoral')
        else:
            styles.append('')
    # Agregar estilos vacíos para las columnas _pred
    styles.extend([''] * len(LABEL_COLUMNS))
    return styles

# Aplicar estilos solo a las columnas de probabilidades
styled_df = df.style.apply(highlight_predictions, axis=1, subset=LABEL_COLUMNS)
styled_df
