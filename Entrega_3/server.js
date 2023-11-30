import express from 'express'
import { exec } from 'child_process';

const app = express()
app.get('/get_result', async (req, res) => {
    try {
        // Comando para ejecutar el script Python
        const comando = 'python simulation.py';

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

            // Puedes usar 'resultadoPython' como desees en tu código de JavaScript
            // por ejemplo, asignarlo a una variable
            const miVariableJS = resultadoPython;
            console.log(`Mi variable de JavaScript: ${miVariableJS}`);
            return res.status(200).json({
                data: resultadoPython
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