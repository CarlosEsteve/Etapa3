## Reto: **"Predicción de Fallos en Equipos en Plantas Energéticas"**

**Objetivo del Reto:**  
Evolucionar la solución para desarrollar un modelo analítico predictivo que permita anticipar fallos en los equipos de la planta energética con una precisión mínima del **80% (accuracy ≥ 0.80)** y por lo menos un **0.75 de f1**. 

Para lograr esto, deberéis:

1. **Mezclar y preprocesar los datos** de los 3 datasets proporcionados.  
2. **Seleccionar variables relevantes** y generar nuevas características si es necesario.  
3. **Construir y evaluar múltiples modelos predictivos**, comparando su rendimiento.  
4. **Ajustar hiperparámetros** y optimizar el modelo para alcanzar la precisión deseada.  
5. **Documentar los hallazgos** y justificar la elección del mejor modelo.  


#### **NOTAS:** 
- Aunque estos datasets tienen naturaleza temporal, se deben tratar las muestras como imputaciones independientes.
- Todas las soluciones en Python deben implementarse mediante funciones. Las funciones deben invocarse al final para ejecutar el proceso deseado.
- Usar el kernel: **Python 3.8 - AzureML (Python 3.10.11)**


### **Estructura de carpetas:**
- Carpeta **src** para los `jupyter notebooks` y `archivos` de trabajo  
- Carpeta **dist** para *exportar los jupyter notebooks a módulos*. (Opcional).
- Ejemplo: 

```
.
└── repositorio/
    ├── dist/
    │   └── [módulos python]
    └── src/
        └── [archivos jupyter y python]
```


### **Trabajo en equipo:**  
   - Usar un repositorio el común.
      - Trabajar en la rama "etapa 3"
   - Trabajo por pares:
      - Recomendable abordar las tareas por pares.
   - Dividir las tareas.
   - Recomendaciones:
      - Evitar trabajar en el mismo archivo a la vez. Especialmente los jupyter notebooks.
      - Comunicarse cuando haya commits y pushes para realizar pulls. Evitar en lo posible conflictos.

### **Entregables:**
- Repositorio que contenga:
1. **Jupyter Notebook conteniendo:  **
   - Código documentado y estructurado.  
   - Visualizaciones clave del análisis exploratorio.  
   - Comparación de modelos con sus métricas.  
   - Justificación del modelo elegido y posibles mejoras.  

2. **Reflexión:** sobre los desafíos encontrados y pasos futuros.

### **Criterios de Evaluación:**  

- **Claridad en el código**: El notebook debe reflejar un proceso estructurado.
- **Feature engineering**: Selección, transformación y generación de variables (según sea necesario).
- **Proceso de entrenamiento**: Selección de candidatos y la secuencia de pasos que garanticen un buen análisis y evaluación de los mismos.
- **Resultado final**: Evidenciar el logro del objetivo planteado y justificar la elección del modelo (si se ha identificado).
