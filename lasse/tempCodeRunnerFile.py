


# # Plot error evolution
# plt.figure(figsize=(10, 6))
# plt.plot(times, l2_errors, 'b-', linewidth=2)
# plt.xlabel('Time')
# plt.ylabel('L2 Error')
# plt.title('Time-dependent Schrödinger Equation: L2 Error Evolution')
# plt.grid(True)
# plt.savefig('schroedinger_error.png', dpi=150, bbox_inches='tight')
# plt.show()


# # In[11]:


# # Compare final solution with exact solution
# t_final = T
# u_exact_final = dolfinx.fem.Function(V, dtype=np.complex128)
# u_exact_final.interpolate(lambda x: exact_solution(x, t_final))


# # In[12]:


# print(f"\nFinal time t={t_final}")
# print(f"Max error (real part): {np.max(np.abs(u_t1.x.array.real - u_exact_final.x.array.real)):.2e}")
# print(f"Max error (imag part): {np.max(np.abs(u_t1.x.array.imag - u_exact_final.x.array.imag)):.2e}")


# # In[13]:


# # Visualization using PyVista
# if MPI.COMM_WORLD.rank == 0:
#     print("\nCreating visualization...")
    
#     try:
#         pyvista.start_xvfb()
        
#         # Create PyVista mesh
#         mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
#         pyvista_cells, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
#         grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)
        
#         # Add solution data
#         grid.point_data["u_real"] = u_t1.x.array.real
#         grid.point_data["u_imag"] = u_t1.x.array.imag
#         grid.point_data["u_magnitude"] = np.abs(u_t1.x.array)
        
#         # Add exact solution data
#         grid.point_data["u_exact_real"] = u_exact_final.x.array.real
#         grid.point_data["u_exact_imag"] = u_exact_final.x.array.imag
#         grid.point_data["u_exact_magnitude"] = np.abs(u_exact_final.x.array)
        
#         # Plot real part of numerical solution
#         grid.set_active_scalars("u_real")
#         p_real = pyvista.Plotter()
#         p_real.add_mesh(grid, show_edges=True)
#         p_real.view_xy()
#         p_real.add_text(f"Real part of numerical solution at t={t_final}", position='upper_left')
#         if not pyvista.OFF_SCREEN:
#             p_real.show(jupyter_backend='none')
#             figure = p_real.screenshot("schroedinger_real.png")
        
#         # Plot imaginary part of numerical solution
#         grid.set_active_scalars("u_imag")
#         p_imag = pyvista.Plotter()
#         p_imag.add_mesh(grid, show_edges=True)
#         p_imag.view_xy()
#         p_imag.add_text(f"Imaginary part of numerical solution at t={t_final}", position='upper_left')
#         if not pyvista.OFF_SCREEN:
#             p_imag.show()
#             figure = p_imag.screenshot("schroedinger_imag.png")
        
#         # Plot magnitude
#         grid.set_active_scalars("u_magnitude")
#         p_mag = pyvista.Plotter()
#         p_mag.add_mesh(grid, show_edges=True)
#         p_mag.view_xy()
#         p_mag.add_text(f"Magnitude of numerical solution at t={t_final}", position='upper_left')
#         if not pyvista.OFF_SCREEN:
#             p_mag.show()
#             figure = p_mag.screenshot("schroedinger_mag.png")
#         print("Visualization completed successfully")
        
#     except Exception as e:
#         print(f"Visualization error: {e}")
#         print("This is normal in containerized environments")


# # In[14]:


# # Grid Spacing Analysis with Interactive 3D Plot
# print("Starting grid spacing convergence study...")

# # Define different grid resolutions to test
# grid_resolutions = [8, 16, 32, 64, 128, 256]  # Reduced for faster computation
# T_study = 1.0  # Time interval
# N_study = 10000   # Time steps
# dt_study = T_study / N_study

# # Storage for results
# results = {}

# for i, nx in enumerate(grid_resolutions):
#     print(f"Solving for grid resolution {nx}x{nx}...")
    
#     # Create mesh for this resolution
#     mesh_study = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)
#     V_study = dolfinx.fem.functionspace(mesh_study, ("Lagrange", 1))
    
#     # Define trial and test functions
#     u_trial = ufl.TrialFunction(V_study)
#     v_test = ufl.TestFunction(V_study)
#     u_prev = dolfinx.fem.Function(V_study, dtype=np.complex128)
    
#     # Set initial condition
#     u_prev.interpolate(initial_condition)
    
#     # Define forms
#     a_study = 1j * ufl.inner(u_trial, v_test) * ufl.dx - dt_study * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
#     L_study = 1j * ufl.inner(u_prev, v_test) * ufl.dx
    
#     # Define boundary conditions
#     mesh_study.topology.create_connectivity(mesh_study.topology.dim-1, mesh_study.topology.dim)
#     boundary_facets_study = dolfinx.mesh.exterior_facet_indices(mesh_study.topology)
#     boundary_dofs_study = dolfinx.fem.locate_dofs_topological(V_study, mesh_study.topology.dim-1, boundary_facets_study)
    
#     u_D_study = dolfinx.fem.Function(V_study, dtype=np.complex128)
#     u_D_study.x.array[:] = 0.0
#     bc_study = dolfinx.fem.dirichletbc(u_D_study, boundary_dofs_study)
    
#     # Create linear problem
#     problem_study = dolfinx.fem.petsc.LinearProblem(a_study, L_study, bcs=[bc_study])
    
#     # Time stepping and error computation
#     times_study = []
#     l2_errors_study = []
    
#     for n in tqdm.tqdm(range(N_study)):
#         t = (n + 1) * dt_study
#         times_study.append(t)
        
#         # Solve
#         u_curr = problem_study.solve()
        
#         # Compute exact solution
#         u_exact_study = dolfinx.fem.Function(V_study, dtype=np.complex128)
#         u_exact_study.interpolate(lambda x: exact_solution(x, t))
        
#         # Compute L2 error
#         error_form = ufl.inner(u_curr - u_exact_study, u_curr - u_exact_study) * ufl.dx
#         error_local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(error_form))
#         error_global = np.sqrt(mesh_study.comm.allreduce(error_local, op=MPI.SUM))
#         l2_errors_study.append(abs(error_global))
        
#         # Update for next time step
#         u_prev.x.array[:] = u_curr.x.array[:]
    
#     # Store results
#     results[nx] = {
#         'times': np.array(times_study),
#         'errors': np.array(l2_errors_study),
#         'h': 1.0/nx
#     }
    
#     print(f"Final error for nx={nx}: {l2_errors_study[-1]:.8e}")

# # Create interactive 3D surface plot
# from mpl_toolkits.mplot3d import Axes3D
# # get_ipython().run_line_magic('matplotlib', 'widget')

# # Prepare data for 3D plot
# Time, H = np.meshgrid(results[grid_resolutions[0]]['times'], 
#                       [1.0/nx for nx in grid_resolutions])

# Error_surface = np.zeros_like(Time)
# for i, nx in enumerate(grid_resolutions):
#     Error_surface[i, :] = results[nx]['errors']

# # Create interactive 3D plot
# fig = plt.figure(figsize=(12, 9))
# ax = fig.add_subplot(111, projection='3d')

# # Plot surface
# surf = ax.plot_surface(Time, H, Error_surface, cmap='viridis', alpha=0.8, 
#                        linewidth=0, antialiased=True)

# # Customize the plot
# ax.set_xlabel('Time', fontsize=12)
# ax.set_ylabel('Grid spacing h', fontsize=12)
# ax.set_zlabel('L2 Error', fontsize=12)
# ax.set_title('L2 Error Evolution: Time × Grid Spacing\n(Interactive - Click and drag to rotate)', fontsize=14)

# # Add colorbar
# cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20)
# cbar.set_label('L2 Error', fontsize=12)

# # Set viewing angle for better initial view
# ax.view_init(elev=20, azim=45)

# # Add grid lines
# ax.grid(True, alpha=0.3)

# # Set z axis to log scale
# # ax.set_zscale('log')

# plt.tight_layout()
# plt.savefig('interactive_3d_error_analysis.png', dpi=150, bbox_inches='tight')
# plt.show()

# # Print summary
# print("\n" + "="*50)
# print("GRID SPACING CONVERGENCE STUDY SUMMARY")
# print("="*50)
# print(f"Time interval: [0, {T_study}]")
# print(f"Time steps: {N_study}")
# print(f"Grid resolutions: {grid_resolutions}")
# print("\nFinal errors:")
# for nx in grid_resolutions:
#     h = 1.0/nx
#     final_err = results[nx]['errors'][-1]
#     print(f"  nx = {nx:3d}, h = {h:.4f}, final error = {final_err:.8e}")

# # Compute convergence rate
# grid_spacings = np.array([1.0/nx for nx in grid_resolutions])
# final_errors = np.array([results[nx]['errors'][-1] for nx in grid_resolutions])
# convergence_rate = np.polyfit(np.log(grid_spacings), np.log(final_errors), 1)[0]
# print(f"\nObserved spatial convergence rate: {convergence_rate:.2f}")
# print("Expected rate for P1 elements: ≈ 2.0")
# print("="*50)


# # In[15]:


# # Two 2D Analysis Plots: Error vs Time and Error vs Grid Spacing
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# # Plot 1: Error vs Time for different grid spacings
# colors = plt.cm.tab10(np.linspace(0, 1, len(grid_resolutions)))

# for i, nx in enumerate(grid_resolutions):
#     times = results[nx]['times']
#     errors = results[nx]['errors']
#     ax1.plot(times, errors, color=colors[i], linewidth=2, 
#              label=f'nx={nx}, h={1.0/nx:.4f}', marker='o', markersize=4)

# ax1.set_xlabel('Time', fontsize=12)
# ax1.set_ylabel('L2 Error', fontsize=12)
# ax1.set_yscale('log')
# ax1.set_title('L2 Error Evolution vs Time\n(Different Grid Spacings)', fontsize=14)
# ax1.legend(fontsize=10)
# ax1.grid(True, alpha=0.3)

# # Plot 2: Error vs Grid Spacing for different times
# # Select specific time points for analysis
# time_indices = [9, 19, 29, 39, 49]  # Every 10th time step
# time_colors = plt.cm.plasma(np.linspace(0, 1, len(time_indices)))

# grid_spacings = [1.0/nx for nx in grid_resolutions]

# for i, time_idx in enumerate(time_indices):
#     errors_at_time = [results[nx]['errors'][time_idx] for nx in grid_resolutions]
#     time_value = results[grid_resolutions[0]]['times'][time_idx]
    
#     ax2.loglog(grid_spacings, errors_at_time, color=time_colors[i], 
#                linewidth=2, marker='s', markersize=6, 
#                label=f't = {time_value:.3f}')

# # Add reference line for O(h²) convergence
# ax2.loglog(grid_spacings, errors_at_time[0] * (np.array(grid_spacings)/grid_spacings[0])**2, 
#            'k--', linewidth=1, alpha=0.7, label='O(h²) reference')

# ax2.set_xlabel('Grid spacing h', fontsize=12)
# ax2.set_ylabel('L2 Error', fontsize=12)
# ax2.set_title('L2 Error vs Grid Spacing\n(Different Times)', fontsize=14)
# ax2.legend(fontsize=10)
# ax2.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig('2d_error_analysis.png', dpi=150, bbox_inches='tight')
# plt.show()

# # Print analysis summary
# print("\n" + "="*60)
# print("2D ERROR ANALYSIS SUMMARY")
# print("="*60)

# print("\nError growth over time:")
# for nx in grid_resolutions:
#     initial_error = results[nx]['errors'][0]
#     final_error = results[nx]['errors'][-1]
#     growth_factor = final_error / initial_error
#     print(f"  nx = {nx:3d}: {initial_error:.2e} → {final_error:.2e} (×{growth_factor:.2f})")

# print(f"\nSpatial convergence at different times:")
# for i, time_idx in enumerate(time_indices):
#     time_value = results[grid_resolutions[0]]['times'][time_idx]
#     errors_at_time = [results[nx]['errors'][time_idx] for nx in grid_resolutions]
    
#     # Compute convergence rate at this time
#     log_h = np.log(grid_spacings)
#     log_error = np.log(errors_at_time)
#     convergence_rate = np.polyfit(log_h, log_error, 1)[0]
    
#     print(f"  t = {time_value:.3f}: convergence rate = {convergence_rate:.2f}")

# print("="*60)


# # In[ ]:




