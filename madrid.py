import streamlit as st # type: ignore
import pandas as pd # type: ignore
import joblib
import numpy as np


# Título de la aplicación
st.title("Predicción de precios de viviendas")
st.header("Ambito: Madrid")
st.write("Ingrese las características de las viviendas")

# Front 

# 2-ciudad: list
df_ciudad = pd.read_csv('distritos.csv', encoding='ISO-8859-1')
opciones_ciudad = df_ciudad['ciudad'].unique().tolist()
#opciones_ciudad =['Madrid', 'Alcalá de Henares', 'Móstoles']
ciudad = st.selectbox('Ciudad:', opciones_ciudad)
#indice_ciudad = opciones_ciudad.index(seleccion_ciudad)
#ciudad = df_ciudad.loc[indice_ciudad, 'ciudad']
#st.write(ciudad)

# 2-distrito: list
df_distrito = df_ciudad[df_ciudad['ciudad'] == ciudad]
df_distrito = df_distrito.reset_index(drop=True)
#st.write(df_distrito)

opciones_distrito = df_distrito['distrito'].tolist()
seleccion_distrito = st.selectbox('Distrito:', opciones_distrito)
indice_barrio = opciones_distrito.index(seleccion_distrito)
distrito = df_distrito.loc[indice_barrio, 'distrito']
distrito_val = df_distrito.loc[indice_barrio, 'precio_m2_distrito']
#st.write(distrito)
#st.write(distrito_val)

# 3-distrito_barrio: list
df_disbar = pd.read_csv('distrito_barrio.csv', encoding='ISO-8859-1')
df_barrio = df_disbar[df_disbar['distrito'] == distrito]
df_barrio = df_barrio.reset_index(drop=True)
#st.write(df_barrio)

#df_barrios = pd.read_csv('barrios.csv')
opciones_barrio = df_barrio['barrio'].tolist()
seleccion_barrio = st.selectbox('Barrio:', opciones_barrio)
indice_barrio = opciones_barrio.index(seleccion_barrio)
barrio = df_barrio.loc[indice_barrio, 'barrio']
barrio_val = df_barrio.loc[indice_barrio, 'precio_m2_barrio']
#st.write(barrio)
#st.write(barrio_val)

# 4-Tipo de vivienda: list
df_vivienda = pd.read_csv('tipo_vivienda.csv')
opciones_vivienda = df_vivienda['tipo_vivienda'].tolist()
seleccion_vivienda = st.selectbox('Tipo de vivienda:', opciones_vivienda)
indice_vivienda = opciones_vivienda.index(seleccion_vivienda)
tipo_vivienda = df_vivienda.loc[indice_vivienda, 'index']
#st.write(tipo_vivienda)

# 1-Metros cuadrados: float
m2 = float(st.text_input("Metros Cuadrados:", value="100"))

# 5-numero de habitaciones: int
#num_habitaciones = int(st.slider('Número de habitaciones :',0 , 8, 1))
num_habitaciones = int(st.text_input("Número de habitaciones:", value="1"))
if num_habitaciones >= 8:
    num_habitaciones = 8

# 6-numero de banos: int
#num_banos = int(st.slider('Número de baños :',1 , 8, 1))
num_banos = int(st.text_input("Número de baños:", value="1"))
if num_banos >=8:
    num_banos = 8

# 7-planta: list
df_planta = pd.read_csv('planta.csv', encoding='ISO-8859-1')
opciones_planta = df_planta['planta'].tolist()
seleccion_planta = st.selectbox('Planta:', opciones_planta)
indice_planta = opciones_planta.index(seleccion_planta)
planta = df_planta.loc[indice_planta, 'valor']
#st.write(planta)

# 8-terraza: bol
options_terraza = ["No", "Si"]
terraza_select = st.selectbox("Tiene terraza:", options_terraza)
terraza_index = options_terraza.index(terraza_select)
terraza = int(terraza_index)

# 9-balcon: bol
options_balcon2 = ["No", "Si"]
balcon_select = st.selectbox("Tiene balcón:", options_balcon2)
balcon_index = options_balcon2.index(balcon_select)
balcon = int(balcon_index)

# 10-ascensor: bol
options_ascensor = ["No", "Si"]
ascensor_select = st.selectbox("Tiene ascensor:", options_ascensor)
ascensor_index = options_ascensor.index(ascensor_select)
ascensor = int(ascensor_index)

# 11-estado_inmmueble: list
df_estado = pd.read_csv('estado_inmueble.csv')
opciones_estado = df_estado['estado'].tolist()
seleccion_estado = st.selectbox('Estado del inmueble:', opciones_estado)
indice_estado = opciones_estado.index(seleccion_estado)
estado = df_estado.loc[indice_estado, 'valor']

## Mostrar la opción seleccionada
#st.write(f'Valor: {estado}')

# Creamos el array de entrada
if (tipo_vivienda != '2') | (tipo_vivienda != '5') | (tipo_vivienda != '6') :
    X_list =    [m2,
                float(distrito_val),
                float(barrio_val),
                int(tipo_vivienda),
                int(num_habitaciones),
                int(num_banos),
                int(planta),
                int(terraza),
                int(balcon),
                int(ascensor),
                int(estado)
                ]
    model = joblib.load('modelo_rf_pisos_joblib.pkl')
else:
    X_list =    [m2,
                float(distrito_val),
                float(barrio_val),
                int(tipo_vivienda),
                int(num_habitaciones),
                int(num_banos),
                ]
    model = joblib.load('modelo_rf_casas_joblib.pkl')

#X = np.array([float(elemento) for elemento in X_list])
X = np.array(X_list, dtype=np.float64)
X = X.reshape(1,-1)

# Botón para ejecutar el modelo
if st.button("Predecir"):
    if len(X) > 0:
        
        # para pisos
        
        #model = joblib.load('modelo_rf_pisos_joblib.pkl')
        
        # Mostrar las primeras filas del DataFrame cargado
        #st.write("Datos cargados:")
        #st.write(X)
        
        #data_scaled = scaler.transform(X)
        
        # Realizar predicciones con el modelo XGBoost
        predicciones = model.predict(X)
        
        # Mostrar las predicciones
        st.write("Predicciones de precio (Euros):")
        #st.write(predicciones)
        corrector = 0.10
        predicciones_bottom = predicciones * (1 - 2*corrector)
        precio_medio = predicciones * (1 - corrector)
        predicciones_top = predicciones
        df = pd.DataFrame({'Precio mínimo':np.round(predicciones_bottom,2),
                           'Precio esperado':np.round(precio_medio,2),
                           'Precio máximo':np.round(predicciones_top,2)})
        st.write(df)
    