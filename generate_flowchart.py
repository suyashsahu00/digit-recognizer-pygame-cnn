import base64
import urllib.request
import urllib.error
import os

# 1. Define configurations for both themes
flowcharts = [
    {
        "name": "dark",
        "output_path": "pipeline_flowchart.png",
        "code": """graph TD
    %% Node Definitions with professional industrial shapes
    A([1. Idle State / Wait for Input])
    B[2. User Draws Stroke on Pygame Canvas]
    C{3. Mouse Button Released?}
    D[4. Calculate Bounding Box of Drawn Coordinates]
    E[5. Capture & Extract Pixel Bounding Box from Canvas]
    F[6. Binarize & Threshold Image Array to Black/White]
    G[7. Pad Image to Square Aspect Ratio]
    H["8. Resize Image to 28x28 Pixels & Normalize to [0, 1]"]
    I["9. Reshape Array to Shape: (1, 28, 28, 1) via NumPy"]
    J[10. Execute Forward Pass on Keras CNN Model]
    K[11. Compute Output Probabilities via Softmax Layer]
    L[12. Extract Class Label Index via np.argmax]
    M[13. Render Label Text and Blit prediction to Pygame DISPLAYSURF]
    N([14. Refresh Screen / Return to Idle])

    %% Flow/Connections
    A -->|Mouse Drag| B
    B --> C
    C -->|No| B
    C -->|Yes| D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> A

    %% Section Subgraphs
    subgraph frontend ["Frontend & UI Layer (Pygame Engine)"]
        A
        B
        C
        M
        N
    end

    subgraph preproc ["Data Processing & Normalization Layer (OpenCV & NumPy)"]
        D
        E
        F
        G
        H
        I
    end

    subgraph model_inference ["Inference & Prediction Layer (Keras CNN)"]
        J
        K
        L
    end

    %% Professional Color Styles (Industrial Slate / Steel / Mint Palette)
    classDef frontStyle fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef preprocStyle fill:#0f172a,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef inferStyle fill:#0f172a,stroke:#a78bfa,stroke-width:2px,color:#f8fafc;

    class A,B,C,M,N frontStyle;
    class D,E,F,G,H,I preprocStyle;
    class J,K,L inferStyle;

    %% Customize subgraph container styling
    style frontend fill:#1e293b,stroke:#38bdf8,stroke-dasharray: 5 5,color:#38bdf8
    style preproc fill:#1e293b,stroke:#34d399,stroke-dasharray: 5 5,color:#34d399
    style model_inference fill:#1e293b,stroke:#a78bfa,stroke-dasharray: 5 5,color:#a78bfa
"""
    },
    {
        "name": "light_printer_friendly",
        "output_path": "pipeline_flowchart_white.png",
        "code": """%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#000000', 'lineColor': '#000000', 'textColor': '#000000', 'fontSize': '14px'}}}%%
graph TD
    %% Node Definitions with professional industrial shapes
    A([1. Idle State / Wait for Input])
    B[2. User Draws Stroke on Pygame Canvas]
    C{3. Mouse Button Released?}
    D[4. Calculate Bounding Box of Drawn Coordinates]
    E[5. Capture & Extract Pixel Bounding Box from Canvas]
    F[6. Binarize & Threshold Image Array to Black/White]
    G[7. Pad Image to Square Aspect Ratio]
    H["8. Resize Image to 28x28 Pixels & Normalize to [0, 1]"]
    I["9. Reshape Array to Shape: (1, 28, 28, 1) via NumPy"]
    J[10. Execute Forward Pass on Keras CNN Model]
    K[11. Compute Output Probabilities via Softmax Layer]
    L[12. Extract Class Label Index via np.argmax]
    M[13. Render Label Text and Blit prediction to Pygame DISPLAYSURF]
    N([14. Refresh Screen / Return to Idle])

    %% Flow/Connections
    A -->|Mouse Drag| B
    B --> C
    C -->|No| B
    C -->|Yes| D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> A

    %% Section Subgraphs (Light grey background, solid black borders)
    subgraph frontend ["Frontend & UI Layer (Pygame Engine)"]
        A
        B
        C
        M
        N
    end

    subgraph preproc ["Data Processing & Normalization Layer (OpenCV & NumPy)"]
        D
        E
        F
        G
        H
        I
    end

    subgraph model_inference ["Inference & Prediction Layer (Keras CNN)"]
        J
        K
        L
    end

    %% Styles for Light/Printer-Friendly Theme (White boxes, black text, black borders)
    classDef default fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000;
    classDef stateStyle fill:#f1f5f9,stroke:#000000,stroke-width:2px,color:#000000;
    classDef decisionStyle fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000;

    class A,N stateStyle;
    class C decisionStyle;

    %% Customize subgraph container styling for light mode
    style frontend fill:#f8fafc,stroke:#000000,stroke-width:2px,stroke-dasharray: 5 5,color:#000000
    style preproc fill:#f8fafc,stroke:#000000,stroke-width:2px,stroke-dasharray: 5 5,color:#000000
    style model_inference fill:#f8fafc,stroke:#000000,stroke-width:2px,stroke-dasharray: 5 5,color:#000000

    %% Thick black arrows (Thick flow line formatting)
    linkStyle default stroke:#000000,stroke-width:3px;
"""
    }
]

# 2. Iterate and render each flowchart
for flow in flowcharts:
    print(f"Generating flowchart diagram for theme: {flow['name']}...")
    
    # Encode the graph definition in URL-safe base64 format
    graph_bytes = flow["code"].encode("utf-8")
    base64_bytes = base64.urlsafe_b64encode(graph_bytes)
    base64_string = base64_bytes.decode("ascii")

    # Fetch from mermaid.ink
    url = f"https://mermaid.ink/img/{base64_string}"
    
    try:
        # Use custom User-Agent to prevent HTTP 403 blocks
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req) as response:
            with open(flow["output_path"], "wb") as f:
                f.write(response.read())
        print(f"Successfully saved {flow['name']} flowchart to '{flow['output_path']}'")
        
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code}")
        try:
            print("Server response:", e.read().decode("utf-8"))
        except Exception as re:
            print("Could not read response:", re)
    except Exception as e:
        print(f"Error fetching the image: {e}")
