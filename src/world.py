#! /usr/bin/env python3
import math
import time
import os

# Tamaño de la esfera
radius = 10
speed = 0.05
theta = 0

# Mapa ASCII simplificado de continentes (latitud x longitud)
# '#' = tierra, ' ' = océano
continents = [
    "       #####             ####        ",
    "     #########        ########       ",
    "    ###########      ##########      ",
    "   ####  ######     ###   ######     ",
    "   ###    ####     ###     ######    ",
    "        ######             #####     ",
    "       #######             #####      ",
    "     ###    ####                       ",
    "    ####      ###                       ",
    "   ####        ###                      ",
    "                                       ",
]

# ANSI colors
BLUE = "\033[34m"   # océano
GREEN = "\033[32m"  # tierra
WHITE = "\033[37m"  # polos/brillo
RESET = "\033[0m"

# Carácteres para sombreado según profundidad
chars = "░▒▓█"

try:
    while True:
        os.system('clear')  # 'cls' en Windows
        for y in range(-radius, radius+1):
            line = ""
            for x in range(-2*radius, 2*radius+1):
                x3d = x / 2  # compensar proporción
                z = math.sqrt(max(radius**2 - x3d**2 - y**2, 0))
                
                # Rotación Y
                xr = x3d * math.cos(theta) + z * math.sin(theta)
                zr = -x3d * math.sin(theta) + z * math.cos(theta)
                
                if xr**2 + y**2 + zr**2 <= radius**2:
                    # Sombreado según profundidad
                    shade_index = int((zr + radius) / (2*radius) * (len(chars)-1))
                    char = chars[shade_index]
                    
                    # Coordenadas de continentes
                    lat = int((y + radius) / (2*radius) * len(continents))
                    lon = int((x + 2*radius) / (4*radius) * len(continents[0]))
                    
                    if 0 <= lat < len(continents) and 0 <= lon < len(continents[0]):
                        if continents[lat][lon] == "#":
                            line += GREEN + char + RESET  # tierra
                        else:
                            line += BLUE + char + RESET   # océano
                    else:
                        line += BLUE + char + RESET
                else:
                    line += " "
            print(line)
        theta += 0.1
        time.sleep(speed)
except KeyboardInterrupt:
    print("\n¡END OF THE WORLD!")








