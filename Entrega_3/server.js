import express from 'express'
import { exec } from 'child_process';

const app = express()
app.get('/get_result', async (req, res) => {
    try {
        // Comando para ejecutar el script Python
        const comando = 'py simulation.py';

        // Ejecutar el comando
        exec(comando, (error, stdout, stderr) => {
            if (error) {
                console.error(`Error: ${error.message}`);
                return;
            }
            if (stderr) {
                console.error(`Error de stderr: ${stderr}`);
                return;
            }

            // Capturar la salida del script Python
            const resultadoPython = stdout.trim();
            console.log(`Resultado de Python: ${resultadoPython}`);

            let parsed_array = JSON.parse(resultadoPython);
            const coord1 = parsed_array[0][0];
            const coord2 = parsed_array[0][0];

            const obstacle1 = parsed_array[1];
            const obstacle2 = parsed_array[2];

            // console.log(JSON.stringify(array, null, 2));
            console.log("array:", parsed_array[1]);

            // console.log(`Mi variable de JavaScript: ${miVariableJS}`);
            return res.status(200).json({
                data: {
                    coord1: coord1,
                    coord2: coord2,
                    obstacle1: obstacle1,
                    obstacle2: obstacle2,
                }
            });
        });
        
    } catch (error){
        return res.status(500).json({
            message: 'Server Error'
        })
    }
})

const port = 3000;
app.listen(port);
console.log(`API corriendo en el puerto ${port}`)