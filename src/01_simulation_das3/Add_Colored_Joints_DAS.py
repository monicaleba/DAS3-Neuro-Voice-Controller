import opensim as osim

def embed_colored_axes_and_labels():
    """
    Embeds colored axis lines into the DAS3 model.
    
    LINE colors = unique per joint (matches the slider legend)
    TIP CUBES   = colored by axis (X=Red, Y=Green, Z=Blue)
    """
    input_model  = r"C:\Users\Lenovo\Desktop\OpenSIM Python\DAS3_release2\OpenSim_model\das3.osim"
    output_model = r"C:\Users\Lenovo\Desktop\OpenSIM Python\DAS3_release2\OpenSim_model\das3_ColoredAxes.osim"
    
    print("Loading original model...")
    model = osim.Model(input_model)
    
    THICKNESS   = 0.0035   # Half-thickness of the rod
    HALF_LENGTH = 0.16     # Half-length of the rod (32cm total)
    TIP_SIZE    = 0.010    # Half-size of the cube tip marker
    
    # --- UPDATED: 10 centimeters of empty space ---
    GAP         = 0.20     
    # ----------------------------------------------

    # Axis tip colors — consistent across ALL joints
    AXIS_TIP_COLORS = {
        "X": (1.0, 0.0, 0.0),   # Red
        "Y": (0.0, 0.8, 0.0),   # Green
        "Z": (0.0, 0.0, 1.0),   # Blue
    }

    # Joint lines: "joint_name": [ ("Axis", (R,G,B) for the LINE color) ]
    JOINT_LINES = {
        "sc1": [("Y", (1.0, 0.2, 0.2))],
        "sc2": [("Z", (1.0, 0.5, 0.0))],
        "sc3": [("X", (1.0, 0.85, 0.0))],
        "ac1": [("Y", (0.0, 0.8, 0.2))],
        "ac2": [("Z", (0.0, 1.0, 0.6))],
        "ac3": [("X", (0.2, 0.9, 0.9))],
        "gh1": [("Y", (0.2, 0.4, 1.0))],
        "gh2": [("Z", (0.5, 0.3, 1.0))],
        "gh3": [("Y", (0.7, 0.2, 1.0))],
        "hu":  [("X", (1.0, 0.0, 0.5))],
        "ur":  [("Y", (0.9, 0.3, 0.6))],
        "rc":  [("X", (0.6, 0.4, 0.2)),
                ("Z", (0.6, 0.4, 0.2))]
    }

    for joint_name, axes_info in JOINT_LINES.items():
        if not model.getJointSet().contains(joint_name):
            print(f"  [SKIP] Joint '{joint_name}' not found")
            continue
            
        joint = model.getJointSet().get(joint_name)
        child_frame = joint.getChildFrame()
        phys_frame = osim.PhysicalFrame.safeDownCast(child_frame)
        
        for axis_index, (axis, line_color) in enumerate(axes_info):
            
            # Add the 10cm GAP to the math so the cube physically separates from the line
            offset_dist = HALF_LENGTH + TIP_SIZE + GAP
            
            if axis == "X":
                half_lengths = osim.Vec3(HALF_LENGTH, THICKNESS, THICKNESS)
                tip_offset   = osim.Vec3(offset_dist, 0, 0)
            elif axis == "Z":
                half_lengths = osim.Vec3(THICKNESS, THICKNESS, HALF_LENGTH)
                tip_offset   = osim.Vec3(0, 0, offset_dist)
            else:  # Y
                half_lengths = osim.Vec3(THICKNESS, HALF_LENGTH, THICKNESS)
                tip_offset   = osim.Vec3(0, offset_dist, 0)
                
            # ── 1. COLORED LINE (Brick) ─────
            brick = osim.Brick(half_lengths)
            brick.setName(f"{joint_name}_line_{axis}_{axis_index}")
            line_app = osim.Appearance()
            line_app.set_color(osim.Vec3(*line_color))
            brick.set_Appearance(line_app)
            child_frame.attachGeometry(brick)

            # ── 2. TIP CUBES ───────
            tip_color = AXIS_TIP_COLORS[axis]
            
            if phys_frame is not None:
                # Positive tip (+axis direction)
                pos_offset = osim.PhysicalOffsetFrame()
                pos_offset.setName(f"tip_{joint_name}_{axis}_pos_{axis_index}")
                pos_offset.setParentFrame(phys_frame)
                pos_offset.set_translation(tip_offset)
                pos_offset.set_orientation(osim.Vec3(0, 0, 0))
                model.addComponent(pos_offset)
                
                tip_cube_pos = osim.Brick(osim.Vec3(TIP_SIZE, TIP_SIZE, TIP_SIZE))
                tip_cube_pos.setName(f"{joint_name}_tip_{axis}_pos_{axis_index}")
                tip_app_pos = osim.Appearance()
                tip_app_pos.set_color(osim.Vec3(*tip_color))
                tip_cube_pos.set_Appearance(tip_app_pos)
                pos_offset.attachGeometry(tip_cube_pos)
                
                # Negative tip (-axis direction)
                neg_tip = osim.Vec3(-tip_offset.get(0), -tip_offset.get(1), -tip_offset.get(2))
                neg_offset = osim.PhysicalOffsetFrame()
                neg_offset.setName(f"tip_{joint_name}_{axis}_neg_{axis_index}")
                neg_offset.setParentFrame(phys_frame)
                neg_offset.set_translation(neg_tip)
                neg_offset.set_orientation(osim.Vec3(0, 0, 0))
                model.addComponent(neg_offset)
                
                tip_cube_neg = osim.Brick(osim.Vec3(TIP_SIZE, TIP_SIZE, TIP_SIZE))
                tip_cube_neg.setName(f"{joint_name}_tip_{axis}_neg_{axis_index}")
                tip_app_neg = osim.Appearance()
                tip_app_neg.set_color(osim.Vec3(*tip_color))
                tip_cube_neg.set_Appearance(tip_app_neg)
                neg_offset.attachGeometry(tip_cube_neg)

    # Finalize and Save
    model.finalizeConnections()
    model.printToXML(output_model)
    
    print(f"\n{'='*60}")
    print(f"  Model saved to: {output_model}")
    print(f"{'='*60}")
    print("  TIP CUBES will now float with a 10cm gap from the lines!")

if __name__ == "__main__":
    embed_colored_axes_and_labels()