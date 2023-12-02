# multiagentes-TC2008B
Repo para almacenar el progreso del reto de la materia

# Equipo 4
Alonso Abimael Morales Reyna A01284747

Ernesto Poisot Ávila A01734765

Marco Ottavio Podesta Vezzali A00833604

Sergio Ortíz Malpica A01284951

# Instructions
0. copy the github repo

1. In the terminal type the following commands
`cd .\Entrega_3\`
`npm i`
`node server.js`

2. Wait until the server is running and listening in port 3000

3. Download the Entrega_3/unity.zip folder and open the project in unity

4. Enter playmode


![Diagramas_01](Entrega_1/Agentes_Eq4_01.jpg "Diagramas de la primera entrega")

## Agentes involucrados
### Combine Harvester
El agente representa un recolector de trigo, capaz de seguir el camino deseado y recolectar el trigo que se encuentre en el camino.
Este agente tiene un limite de capacidad de alamacenamiento de trigo y gasolina.
Cada paso que hace, consume gasolina.
Cada trigo que recoge, es acumulado en su capacidad de carga.
Si se llega a una capacidad de carga mayor al 75%, o su capacidad de gasolina se reduce del 25%, este se detiene para esperar al tractor recolector.

### Truck
El agente representa un tractor recolector de lo que haya procesado el Combine. Además, ofrece la posibilidad de descargar el combine y rellenar su gasolina.
El agente tiene un limite de capacidad de almacenamiento de trigo y gasolina.
Cada paso que hace, consume gasolina.
Si se llega a una capacidad de carga mayor al 75%, o su capacidad de gasolina se reduce del 25%, este se detiene para esperar al tractor recolector.
El agente está en un monitoreo constante de los estados del Combine, si determina que el Combine necesita ayuda, este va a ir a hacer las acciones necesarias (descargar trigo, rellenar gasolina).

