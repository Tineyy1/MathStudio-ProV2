import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import plotly.graph_objects as go
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="MathStudio Pro", page_icon="📐", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📐 MathStudio Pro")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Select Module", ["Root Finding Analysis", "Advanced Matrix Operations"])
st.sidebar.markdown("---")
st.sidebar.info("Developed with Streamlit & Python")

## ==========================================
# MODULE 1: ROOT FINDING
# ==========================================
if app_mode == "Root Finding Analysis":
    st.title("Root Finding Analysis")
    st.markdown("Analyze equations and find roots using numerical methods with interactive visualizations.")
    
    col_input, col_results = st.columns([1, 2.5])
    
    with col_input:
        st.subheader("Parameters")
        eq_str = st.text_input("Equation f(x)", value="3*x + sin(x) - exp(x)", help="Use standard Python math (e.g. x**3 or x^3, sin(x), exp(x))")
        method = st.selectbox("Algorithm", ["Incremental Search", "Bisection Method", "Regula-Falsi", "Newton-Raphson", "Secant Method"])
        
        # Dynamic inputs based strictly on the provided documents
        if method == "Incremental Search":
            xl = st.number_input("Initial Value (xl)", value=0.0, format="%.4f")
            delta_x = st.number_input("Initial Increment (Δx)", value=0.5, format="%.4f")
        elif method in ["Bisection Method", "Regula-Falsi"]:
            xl = st.number_input("Lower Bound (xl)", value=-0.5 if method == "Regula-Falsi" else 0.4, format="%.4f")
            xu = st.number_input("Upper Bound (xu)", value=1.0 if method == "Regula-Falsi" else 0.6, format="%.4f")
        elif method == "Newton-Raphson":
            x0 = st.number_input("Initial Guess (xi)", value=-5.0, format="%.4f")
        elif method == "Secant Method":
            x_prev = st.number_input("First Guess (x_i-1)", value=0.5, format="%.4f")
            x0 = st.number_input("Second Guess (x_i)", value=5.0, format="%.4f")
            
        tol = st.number_input("Tolerance (Stopping Criterion)", value=0.001, format="%.5f")
        max_iter = st.number_input("Max Iterations", value=50, step=1)
        solve_btn = st.button("Calculate Root")

    with col_results:
        if solve_btn:
            try:
                x = sp.Symbol('x')
                safe_eq_str = eq_str.replace('^', '**')
                expr = sp.sympify(safe_eq_str)
                f = sp.lambdify(x, expr, 'numpy')
                df = sp.lambdify(x, sp.diff(expr, x), 'numpy')

                results, root, iterations, final_err = [], None, 0, 0
                
                # --- 1. INCREMENTAL SEARCH ---
                if method == "Incremental Search":
                    curr_xl = xl
                    curr_dx = delta_x
                    
                    for i in range(max_iter):
                        curr_xu = curr_xl + curr_dx
                        fxl, fxu = f(curr_xl), f(curr_xu)
                        prod = fxl * fxu
                        
                        if prod > 0:
                            remark = "Go to next interval"
                        else:
                            remark = "Revert back to xl & consider smaller interval"
                            
                        # Matching PDF Table Columns
                        results.append({
                            "Iteration": i + 1,
                            "x_l": curr_xl,
                            "Δx": curr_dx,
                            "x_u": curr_xu,
                            "f(x_l)": fxl,
                            "f(x_u)": fxu,
                            "f(x_l)*f(x_u)": "> 0" if prod > 0 else "< 0",
                            "Remark": remark
                        })
                        
                        if abs(fxu) < tol or curr_dx < (tol / 10):
                            root, iterations = curr_xu, i + 1
                            break
                            
                        if prod > 0:
                            curr_xl = curr_xu  # Go to next interval
                        else:
                            curr_dx = curr_dx / 10.0  # Reduce interval step
                            
                # --- 2. BISECTION METHOD ---
                elif method == "Bisection Method":
                    if f(xl) * f(xu) > 0:
                        st.warning("f(xl) and f(xu) must have opposite signs for Bisection.")
                        st.stop()
                        
                    xr_old = None
                    for i in range(max_iter):
                        xr = (xl + xu) / 2
                        fxl, fxr = f(xl), f(xr)
                        prod = fxl * fxr
                        
                        ea = abs((xr - xr_old) / xr) * 100 if xr_old is not None else None
                        
                        if prod < 0:
                            remark = "1st subinterval"
                        else:
                            remark = "2nd subinterval"
                            
                        # Matching PDF Table Columns
                        results.append({
                            "Iteration": i + 1,
                            "x_l": xl,
                            "x_r": xr,
                            "x_u": xu,
                            "f(x_l)": fxl,
                            "f(x_r)": fxr,
                            "|E_a| %": ea if ea is not None else "",
                            "f(x_l)*f(x_r)": "< 0" if prod < 0 else "> 0",
                            "Remark": remark
                        })
                        
                        if (ea is not None and ea < tol) or fxr == 0:
                            root, iterations, final_err = xr, i + 1, ea
                            break
                            
                        if prod < 0:
                            xu = xr
                        else:
                            xl = xr
                        xr_old = xr

                # --- 3. REGULA-FALSI (FALSE POSITION) ---
                elif method == "Regula-Falsi":
                    if f(xl) * f(xu) > 0:
                        st.warning("f(xl) and f(xu) must have opposite signs for Regula-Falsi.")
                        st.stop()
                        
                    xr_old = None
                    for i in range(max_iter):
                        fxl, fxu = f(xl), f(xu)
                        if fxl - fxu == 0:
                            st.error("Division by zero encountered.")
                            break
                            
                        xr = (xu * fxl - xl * fxu) / (fxl - fxu)
                        fxr = f(xr)
                        prod = fxl * fxr
                        
                        ea = abs((xr - xr_old) / xr) if xr_old is not None else None
                        
                        # Matching PDF Table Columns
                        results.append({
                            "No. of Iteration": i + 1,
                            "x_L": xl,
                            "x_U": xu,
                            "x_R": xr,
                            "E_a": ea if ea is not None else "",
                            "f(x_L)": fxl,
                            "f(x_U)": fxu,
                            "f(x_R)": fxr,
                            "f(x_L)*f(x_R)": "< 0" if prod < 0 else "> 0"
                        })
                        
                        if (ea is not None and ea < tol) or fxr == 0:
                            root, iterations, final_err = xr, i + 1, ea
                            break
                            
                        if prod < 0:
                            xu = xr
                        else:
                            xl = xr
                        xr_old = xr

                # --- 4. NEWTON-RAPHSON ---
                elif method == "Newton-Raphson":
                    xi = x0
                    # Initial state (Iteration 0) matching PDF format
                    results.append({
                        "No. of iteration": 0,
                        "x_i": xi,
                        "E_a": "",
                        "f(x)": f(xi),
                        "f'(x)": df(xi)
                    })
                    
                    for i in range(max_iter):
                        fxi, dfxi = f(xi), df(xi)
                        if dfxi == 0:
                            st.error("Derivative became zero. Newton-Raphson fails.")
                            break
                            
                        xi_new = xi - fxi / dfxi
                        ea = abs((xi_new - xi) / xi_new)
                        
                        xi = xi_new
                        # Matching PDF Table Columns
                        results.append({
                            "No. of iteration": i + 1,
                            "x_i": xi,
                            "E_a": ea,
                            "f(x)": f(xi),
                            "f'(x)": df(xi)
                        })
                        
                        if ea < tol:
                            root, iterations, final_err = xi, i + 1, ea
                            break

                # --- 5. SECANT METHOD ---
                elif method == "Secant Method":
                    xi_prev, xi = x_prev, x0
                    for i in range(max_iter):
                        fxi, fxi_prev = f(xi), f(xi_prev)
                        if fxi - fxi_prev == 0:
                            st.error("Difference between f(x_i) and f(x_i-1) is zero. Secant fails.")
                            break
                            
                        xi_new = xi - (fxi * (xi - xi_prev)) / (fxi - fxi_prev)
                        ea = abs((xi_new - xi) / xi_new)
                        
                        # Matching Word Document Table Columns
                        results.append({
                            "Iteration Number": i + 1,
                            "x_{i-1}": xi_prev,
                            "x_i": xi,
                            "x_{i+1}": xi_new,
                            "E_a": ea,
                            "f(x_{i-1})": fxi_prev,
                            "f(x_i)": fxi,
                            "f(x_{i+1})": f(xi_new)
                        })
                        
                        xi_prev, xi = xi, xi_new
                        if ea < tol:
                            root, iterations, final_err = xi_new, i + 1, ea
                            break

                # --- METRICS & GRAPH ---
                if len(results) > 0:
                    if root is not None:
                        st.toast('Calculation Complete!', icon='✅')
                    else:
                        st.toast('Max iterations reached or stopped.', icon='⚠️')
                        # Extract the best guess for graph centering
                        latest_row = results[-1]
                        root = latest_row.get("x_u", latest_row.get("x_R", latest_row.get("x_i", latest_row.get("x_{i+1}", 0))))
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Calculated Root", f"{root:.6f}")
                    m2.metric("Total Iterations", iterations if iterations > 0 else max_iter)
                    m3.metric("Final Error (E_a)", f"{final_err:.6f}" if final_err else "N/A")
                    st.divider()

                    # Interactive Plotly Graph
                    x_vals = np.linspace(float(root) - 3, float(root) + 3, 300)
                    y_vals = f(x_vals)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='f(x)', line=dict(color='royalblue', width=2)))
                    fig.add_hline(y=0, line_dash="dash", line_color="black")
                    fig.add_vline(x=0, line_dash="dash", line_color="black")
                    fig.add_trace(go.Scatter(x=[root], y=[0], mode='markers', name='Root', marker=dict(color='red', size=12, symbol='x')))
                    
                    fig.update_layout(title="Interactive Function Graph", xaxis_title="X Axis", yaxis_title="Y Axis", hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                    # Dynamic Expander Table - Pandas automatically reads the dictionaries and sets the exact headers
                    with st.expander("📊 View Detailed Iteration History", expanded=True):
                        st.dataframe(pd.DataFrame(results), use_container_width=True)

            except Exception as e:
                st.error(f"Error evaluating equation or solving. Details: {e}")
# ==========================================
# MODULE 2: MATRIX OPERATIONS
# ==========================================
elif app_mode == "Advanced Matrix Operations":
    st.title("Advanced Matrix Operations")
    st.markdown("Use the interactive spreadsheets below to input your matrix data.")
    
    op = st.selectbox("Select Operation", ["Addition", "Multiplication", "System of Equations (Ax = B)", "Adjoint", "Inverse", "Determinant", "Power of Matrix", "Transpose"])
    
    st.divider()
    
    # Dynamic Layout based on Operation
    if op in ["Addition", "Multiplication", "System of Equations (Ax = B)"]:
        col1, col2 = st.columns(2)
    else:
        col1, col2 = st.columns([1, 1]) # Keep it centered
        
    with col1:
        st.subheader("Matrix A")
        c1, c2 = st.columns(2)
        rows_A = c1.number_input("Rows A", 1, 10, 3)
        cols_A = c2.number_input("Cols A", 1, 10, 3)
        
        # INTERACTIVE DATA EDITOR FOR A
        df_A = pd.DataFrame(np.zeros((rows_A, cols_A)), columns=[f"Col {i+1}" for i in range(cols_A)])
        edited_A = st.data_editor(df_A, use_container_width=True, key="matrix_a")
        A = edited_A.to_numpy()

    if op in ["Addition", "Multiplication", "System of Equations (Ax = B)"]:
        with col2:
            st.subheader("Matrix B")
            if op == "System of Equations (Ax = B)":
                st.info("Matrix B must be a single column (Results vector)")
                rows_B = rows_A
                cols_B = 1
            elif op == "Addition":
                rows_B, cols_B = rows_A, cols_A
            else: # Multiplication
                c1, c2 = st.columns(2)
                rows_B = c1.number_input("Rows B", 1, 10, cols_A, disabled=True)
                cols_B = c2.number_input("Cols B", 1, 10, 3)
                
            # INTERACTIVE DATA EDITOR FOR B
            df_B = pd.DataFrame(np.zeros((rows_B, cols_B)), columns=[f"Col {i+1}" for i in range(cols_B)])
            edited_B = st.data_editor(df_B, use_container_width=True, key="matrix_b")
            B = edited_B.to_numpy()
            
    if op == "Power of Matrix":
        with col2:
            st.subheader("Settings")
            power = st.number_input("Calculate to the power of (n):", value=2, step=1)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Execute Matrix Operation", use_container_width=True):
        try:
            with st.spinner("Calculating..."):
                time.sleep(0.5) # Slight delay for professional UX feel
                st.subheader("Resulting Matrix")
                
                if op == "Addition":
                    result = A + B
                elif op == "Multiplication":
                    result = np.matmul(A, B)
                elif op == "Transpose":
                    result = A.T
                elif op == "Determinant":
                    result = np.linalg.det(A)
                    st.metric("Determinant Value", f"{result:.4f}")
                    result = None # Skip table
                elif op == "Inverse":
                    result = np.linalg.inv(A)
                elif op == "Adjoint":
                    result = np.linalg.inv(A) * np.linalg.det(A)
                    result = np.round(result, 4)
                elif op == "Power of Matrix":
                    result = np.linalg.matrix_power(A, power)
                elif op == "System of Equations (Ax = B)":
                    result = np.linalg.solve(A, B)
                    st.success("Solutions found for Vector X:")
                    
                # Display output beautifully
                if result is not None:
                    st.dataframe(pd.DataFrame(result), use_container_width=True)
                    st.toast('Operation Successful!', icon='✅')
                    
        except np.linalg.LinAlgError as e:
            st.error(f"Mathematical Error: {e} (e.g., Matrix might be singular/non-invertible)")
        except ValueError as e:
            st.error(f"Dimension Error: {e}")