# NASDAQ Stock Analysis App

Nasdaq stock -V3 : se centra en mejorar las recomendaciones de acciones profundizando en la estrategia con el indicador avanzado Nadaraya-Watson Envelope. elimina también el codgio duplicado y funciona correctamente. Se considera muy útil para el seguimiento y análisis del grupo de acciones elegidas.

Incluye un documento llamado stock_analyzer que se puede utilizar para aplicar este análisis y estrategia de recomendaciones sobre otro programa o una base de datos





Esta aplicación proporciona análisis en tiempo real de las 7 principales empresas del NASDAQ, incluyendo predicciones y recomendaciones de trading basadas en análisis técnico.

## Características

- Análisis técnico en tiempo real
- Predicciones para el próximo día de trading
- Datos del pre-mercado
- Indicadores técnicos avanzados
- Recomendaciones automatizadas de trading

## Requisitos

- Docker
- Docker Compose

## Instalación y Despliegue

1. Clonar el repositorio:
```bash
git clone <tu-repositorio>
cd <directorio-del-proyecto>
```

2. Construir y ejecutar con Docker Compose:
```bash
docker-compose up -d --build
```

3. Acceder a la aplicación:
- Abrir en el navegador: `http://<ip-del-servidor>:8501`
- Para acceso local: `http://localhost:8501`

## Acceso Remoto

Para acceder desde otros ordenadores en la red:
1. Asegúrate de que el puerto 8501 está abierto en el firewall
2. Usa la IP del servidor donde está desplegada la aplicación
3. Accede mediante: `http://<ip-del-servidor>:8501`

## Mantenimiento

- Para ver los logs:
```bash
docker-compose logs -f
```

- Para reiniciar la aplicación:
```bash
docker-compose restart
```

- Para detener la aplicación:
```bash
docker-compose down
```

## Actualizaciones

Para actualizar la aplicación:
```bash
git pull
docker-compose up -d --build
```

## Notas de Seguridad

- La aplicación está configurada para ejecutarse en una red local
- Para exposición a Internet, se recomienda añadir un proxy inverso (como Nginx) con SSL
- Considerar añadir autenticación si se expone a Internet

## Soporte de Zona Horaria

La aplicación está configurada para usar la zona horaria de Nueva York (ET) para sincronización con el mercado de valores.

## Disclaimer

Esta aplicación proporciona análisis técnico y recomendaciones automatizadas. Las decisiones de trading deben tomarse bajo su propia responsabilidad. 
