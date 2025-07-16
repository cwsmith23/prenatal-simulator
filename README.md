 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/README.md b/README.md
index 501282a98f028ad17f0da7dc5288d2120168c2c3..d1be099d6a59316ca875127df00e5ba3504472d0 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,21 @@
-# prenatal-simulator
+# prenatal-simulator
+
+## Requirements
+
+- Python 3.10 or newer
+
+## Installation
+
+Install the dependencies using:
+
+```bash
+pip install -r requirements.txt
+```
+
+## Usage
+
+Run the Streamlit app with:
+
+```bash
+streamlit run app.py
+```
 
EOF
)
