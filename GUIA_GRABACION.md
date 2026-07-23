# Guía de grabación de tiendas para reconstrucción 3D

Para quien graba: este video NO es un video de marketing — es la materia prima
de un modelo 3D. La cámara reemplaza los ojos de un escáner: lo que no se ve
(o se ve borroso) simplemente no existirá en el modelo. Seguir esta guía al
pie de la letra es la diferencia entre un modelo perfecto y uno inservible.

## 1. Configuración de la cámara (una sola vez, antes de grabar)

**Regla de oro para ambos casos: TODO en manual y bloqueado.** Nada en "auto".
Cada parámetro automático (foco, exposición, balance) cambia durante la
grabación y el sistema 3D asume una cámara de parámetros fijos.

### Caso A — iPhone / teléfono

Usar la app **Blackmagic Camera** (gratis, iPhone/Android) — NO la cámara nativa.

| Ajuste | Valor |
|---|---|
| Resolución / fps | 4K (3840×2160) a **30 o 60 fps** |
| Códec | HEVC calidad alta (no ProRes) |
| Rango dinámico | **SDR / Rec.709** — HDR APAGADO |
| Estabilización | **APAGADA** |
| Lente | **1x principal** — NUNCA cambiar de lente durante un video |
| Enfoque | **Manual, fijo** a ~2–3 metros |
| Obturación | **1/100 fija** (evita parpadeo de luces en Chile y motion blur) |
| ISO | Manual, el que dé buena exposición (típico 200–800) |
| Balance de blancos | Fijo (~4500–5000 K) |

### Caso B — Cámara profesional (mirrorless / cine)

| Ajuste | Valor |
|---|---|
| Resolución / fps | 4K (UHD) a **30 o 60 fps** |
| Códec / perfil | H.264/HEVC 8-bit 4:2:0 estándar. Perfil de color **Rec.709 normal** — NADA de S-Log/V-Log/HLG/RAW (el sistema necesita gamma estándar, no material para gradear) |
| Estabilización | **TODA apagada**: IBIS (del cuerpo) + OIS (del lente) + digital |
| Lente | **Focal fija** durante todo el video: prime gran angular o zoom fijado en el extremo ancho (16–24 mm equivalente). NO zoomear jamás mientras se graba |
| Enfoque | **Manual, fijo** a ~2–3 m. Desactivar AF continuo. Si el lente tiene "focus breathing compensation", apagarla |
| Obturación | **1/100 fija** (red de 50 Hz — nunca 1/60 ni 1/125). Ignorar la regla cinematográfica de 180° — aquí queremos frames congelados, no motion blur bonito |
| Apertura | Fija, **f/4–f/5.6** (profundidad de campo amplia; nada de f/1.8 con fondos desenfocados) |
| ISO | Manual, fijo, el que cierre la exposición (el ruido de ISO 1600 de un sensor grande es aceptable; el blur no) |
| Balance de blancos | Fijo en Kelvin (~4500–5000 K), no AWB |
| Audio | Irrelevante — puede ir apagado |

Notas para el caso B:
- Un sensor grande da mejor nitidez y menos ruido que un teléfono — es la
  mejor opción si el operador respeta la técnica de la sección 2.
- Peso: grabar a pulso firme con correa/rig de dos manos. NO usar gimbal con
  modo "follow" agresivo (suaviza rotaciones de forma no física); un gimbal
  en modo lock/estático suave es aceptable pero innecesario si se camina lento.
- Si la cámara ofrece obturador electrónico con rolling shutter fuerte,
  preferir el modo de lectura rápida o el obturador mecánico.

### Verificación rápida (ambos casos)

Antes de grabar en serio: grabar 10 segundos caminando, pausar el video y
verificar que un fotograma congelado se vea **nítido** (sin barrido). Si se
ve movido, bajar la velocidad al caminar o abrir más la obturación no — la
obturación no se toca: caminar más lento.

## 2. Técnica de grabación (lo más importante)

1. **Teléfono horizontal**, sostenido con las dos manos a la altura del pecho,
   apuntando ligeramente hacia adelante (no al piso ni al techo).
2. **Caminar LENTO** — la mitad de la velocidad que se siente natural.
3. **Micro-pausas**: cada 2–3 pasos, detenerse medio segundo. Estas pausas
   generan los fotogramas perfectos que usa el sistema.
4. **Nunca girar en el lugar.** Los giros siempre caminando, describiendo una
   curva amplia. La regla: si los pies no se mueven, la cámara tampoco gira.
5. **Trayectorias con desplazamiento lateral**: recorrer los pasillos por el
   centro mirando hacia un lado, volver mirando hacia el otro. El modelo 3D
   se construye con el movimiento lateral (paralaje).
6. **Cerrar bucles**: terminar el recorrido volviendo al punto de partida, y
   cruzar entre pasillos/zonas varias veces. Cada zona repasada "ancla" el
   mapa y evita deformaciones.
7. **Traslape entre zonas**: al pasar de una zona a otra, grabar la
   transición caminando (no cortar el video justo en el límite).
8. Acercarse a **1.5–2.5 m** de los productos/muebles que importan; evitar
   pasar a menos de 1 m de superficies grandes y lisas (paredes, puertas de
   clóset) — llenan la pantalla sin dar información.
9. **Sin gente en cuadro** en lo posible (grabar antes de abrir o en horas
   valle). Las personas en movimiento generan "fantasmas" en el modelo.
10. Luces de la tienda **todas encendidas**, siempre las mismas.

## 3. Duración y cobertura

La duración correcta la determina la SUPERFICIE de la tienda y el paso lento,
no un número fijo. Regla práctica: **~2 minutos por cada 100 m²** de
superficie recorrible, con un **mínimo de 2 minutos** por video.

| Tienda / zona | Duración de referencia |
|---|---|
| Zona chica o box (~100 m²) | 2–3 min |
| Tienda mediana (~300 m²) | 6–8 min |
| Tienda grande (~600 m²) | 12–15 min, idealmente 2–3 videos por zonas |
| Muy grande (>1000 m²) | un video por zona (con traslape entre zonas) |

- A 60 fps, un minuto de video son 3.600 fotogramas — sobra material; lo que
  importa es que el RECORRIDO cubra todo, no que el video sea largo.
- **Mejor varios videos por zona que uno eterno**: más fáciles de repetir si
  algo sale mal, y el sistema los une siempre que haya traslape entre ellos
  (empezar cada video en una zona ya cubierta por el anterior).
- Espacio: 4K60 HEVC ocupa ~0.5–1 GB por minuto. Llevar el teléfono con
  ≥50 GB libres y batería completa.

## 4. Errores que arruinan la grabación (aprendidos a golpes)

- ❌ HDR activado o cámara nativa del iPhone con estabilización → modelo roto.
- ❌ Girar en el lugar / paneos rápidos → zonas irrecuperables.
- ❌ Video vertical → menos campo visual, peor conexión entre fotogramas.
- ❌ Cambiar de lente (0.5x ↔ 1x) a mitad de video.
- ❌ Caminar rápido → motion blur → modelo borroso.
- ❌ Foco/exposición en automático → parámetros inestables.
- ❌ Cortar el video en medio de una zona sin traslape con el siguiente.

## 5. Entrega

Subir los archivos originales (sin recomprimir, sin pasar por WhatsApp) a la
carpeta compartida, un directorio por tienda:
`<tienda>/<fecha>/A001.mov, A002.mov, ...` + una nota con el orden del
recorrido y un plano/croquis marcando la ruta si es posible.
