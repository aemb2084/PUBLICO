# Versión SIMPLE - Sin estilos visuales

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
df_probs = pd.DataFrame(all_preds, columns=LABEL_COLUMNS)
df_preds = pd.DataFrame((all_preds >= THRESHOLD).astype(int), columns=[f'{col}_pred' for col in LABEL_COLUMNS])
df_questions = pd.DataFrame({'pregunta': test_comment})

# Combinar todo
df_results = pd.concat([df_questions, df_probs, df_preds], axis=1)

# Guardar resultados en CSV
output_file = f"{LOGS_DIR}/predicciones_test.csv"
df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n✅ Resultados guardados en: {output_file}")

# Mostrar resumen
print("\n=== RESUMEN DE PREDICCIONES ===")
print("\nProbabilidades (valores 0-1):")
print(df_probs.round(3))

print("\nPredicciones binarias (threshold={}):" .format(THRESHOLD))
print(df_preds)

# Análisis por pregunta
print("\n=== ANÁLISIS POR PREGUNTA ===")
for i, question in enumerate(test_comment):
    predicted_labels = [LABEL_COLUMNS[j] for j in range(len(LABEL_COLUMNS)) if all_preds[i][j] >= THRESHOLD]
    print(f"\n{i+1}. {question[:80]}...")
    print(f"   Predichas: {', '.join(predicted_labels) if predicted_labels else 'Ninguna'}")
    print(f"   Probabilidades: {dict(zip(LABEL_COLUMNS, all_preds[i].round(3)))}")

# Mostrar DataFrame completo
df_results
