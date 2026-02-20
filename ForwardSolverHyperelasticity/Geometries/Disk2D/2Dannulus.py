import gmsh
import sys
import math

# Initialize GMSH
gmsh.initialize(sys.argv)
    
# Create a new model
gmsh.model.add("annulus_simple")

def create_annulus_simple(inner_radius=0.5, outer_radius=1.0, center_x=0.0, center_y=0.0,
                         mesh_size=0.1, export_mesh=True, filename="annulus.msh"):
    """
    Alternative method using disk with hole.
    This is often simpler for creating annular regions.
    """
    
    
    # Create outer disk
    outer_disk = gmsh.model.occ.addDisk(center_x, center_y, 0, outer_radius, outer_radius)
    
    # Create inner disk (hole)
    inner_disk = gmsh.model.occ.addDisk(center_x, center_y, 0, inner_radius, inner_radius)
    
    # Cut the inner disk from the outer disk to create annulus
    annulus = gmsh.model.occ.cut([(2, outer_disk)], [(2, inner_disk)])


    #p1 = gmsh.model.occ.addPoint(inner_radius, 0.0, 0.0, mesh_size)  # Right point
    #p2 = gmsh.model.occ.addPoint(outer_radius, 0.0, 0.0, mesh_size)  # Top point

    
    # Synchronize the model
    gmsh.model.occ.synchronize()
    
    # Get the resulting surface (annulus)
    surfaces = annulus[0]
    if surfaces:
        annulus_surface = surfaces[0][1]
        
        # Add physical groups
        gmsh.model.addPhysicalGroup(2, [annulus_surface], tag=1)
        gmsh.model.setPhysicalName(2, 1, "Annulus")
    
    # # Find boundary curves
    # boundaries = gmsh.model.getBoundary([(2, annulus_surface)], oriented=False)
    
    # # Separate inner and outer boundaries
    # outer_boundaries = []
    # inner_boundaries = []
    
    # for boundary in boundaries:
    #     # Get the center of the curve to determine if it's inner or outer
    #     com = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
    #     distance = math.sqrt((com[0] - center_x)**2 + (com[1] - center_y)**2)
        
    #     if abs(distance - outer_radius) < 1e-6:
    #         outer_boundaries.append(boundary[1])
    #     elif abs(distance - inner_radius) < 1e-6:
    #         inner_boundaries.append(boundary[1])
    
    # # Add physical groups for boundaries
    # if outer_boundaries:
    #     gmsh.model.addPhysicalGroup(1, outer_boundaries, tag=2)
    #     gmsh.model.setPhysicalName(1, 2, "OuterBoundary")
    
    # if inner_boundaries:
    #     gmsh.model.addPhysicalGroup(1, inner_boundaries, tag=3)
    #     gmsh.model.setPhysicalName(1, 3, "InnerBoundary")
    
    # Set mesh size
    #gmsh.model.addPhysicalGroup(0, [p1], tag=10)
    #gmsh.model.addPhysicalGroup(0, [p2], tag=20)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    
    # Generate 2D mesh
    gmsh.model.mesh.generate(2)
    
    # Optimize mesh
    gmsh.model.mesh.optimize("Laplace2D")
    
    # Export mesh if requested
    if export_mesh:
        gmsh.write(filename)
        print(f"Mesh saved to {filename}")
    
    # Launch GUI
    #gmsh.fltk.run()
    
    # Finalize GMSH
    #gmsh.finalize()


if __name__ == "__main__":
    # Example usage 1: Using the detailed method
    print("Creating annulus using detailed method...")
    create_annulus_simple(
        inner_radius=0.5,
        outer_radius=1.0,
        mesh_size=0.0075,
        export_mesh=True,
        filename="annulus_2D.msh"
    )
    
    # Example usage 2: Using the simple method (uncomment to use)
    # print("\nCreating annulus using simple method...")
    # create_annulus_simple(
    #     inner_radius=0.3,
    #     outer_radius=1.0,
    #     mesh_size=0.05,
    #     export_mesh=True,
    #     filename="annulus_simple.msh"
    # )