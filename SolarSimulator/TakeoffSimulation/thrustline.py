"""
AeroSandbox Thrustline Position Investigation
Analyzes the effect of moving the thrustline above the wing on aircraft trim and performance
"""

import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt

# Define a simple aircraft geometry
def create_aircraft(thrustline_height):
    """
    Create an aircraft with specified thrustline height above the wing
    
    Parameters:
    -----------
    thrustline_height : float
        Height of thrustline above wing chord reference (meters)
    """
    
    # Define wing
    wing = asb.Wing(
        name="Main Wing",
        symmetric=True,
        xsecs=[
            asb.WingXSec(
                xyz_le=[0, 0, 0],
                chord=1.0,
                airfoil=asb.Airfoil("naca2412"),
            ),
            asb.WingXSec(
                xyz_le=[0.1, 5.0, 0.5],
                chord=0.6,
                airfoil=asb.Airfoil("naca2412"),
            ),
        ]
    )
    
    # Define horizontal tail
    hstab = asb.Wing(
        name="Horizontal Stabilizer",
        symmetric=True,
        xsecs=[
            asb.WingXSec(
                xyz_le=[5.0, 0, 0.1],
                chord=0.5,
                airfoil=asb.Airfoil("naca0012"),
            ),
            asb.WingXSec(
                xyz_le=[5.1, 1.5, 0.1],
                chord=0.3,
                airfoil=asb.Airfoil("naca0012"),
            ),
        ]
    )
    
    # Define vertical tail
    vstab = asb.Wing(
        name="Vertical Stabilizer",
        symmetric=False,
        xsecs=[
            asb.WingXSec(
                xyz_le=[5.0, 0, 0.1],
                chord=0.6,
                airfoil=asb.Airfoil("naca0012"),
            ),
            asb.WingXSec(
                xyz_le=[5.3, 0, 1.2],
                chord=0.3,
                airfoil=asb.Airfoil("naca0012"),
            ),
        ]
    )
    
    # Create fuselage
    fuselage = asb.Fuselage(
        name="Fuselage",
        xsecs=[
            asb.FuselageXSec(xyz_c=[0, 0, 0], radius=0.3),
            asb.FuselageXSec(xyz_c=[1, 0, 0], radius=0.4),
            asb.FuselageXSec(xyz_c=[4, 0, 0], radius=0.4),
            asb.FuselageXSec(xyz_c=[5.5, 0, 0], radius=0.1),
        ]
    )
    
    # Create airplane with specified thrustline
    airplane = asb.Airplane(
        name=f"Aircraft (thrustline: {thrustline_height:.2f}m)",
        wings=[wing, hstab, vstab],
        fuselages=[fuselage],
        xyz_ref=[0.3, 0, 0],  # CG location
    )
    
    return airplane, thrustline_height


def analyze_trim_with_thrust(airplane, thrustline_height, velocity=25, thrust=100):
    """
    Analyze aircraft trim condition with thrust effects
    
    Parameters:
    -----------
    airplane : asb.Airplane
        Aircraft geometry
    thrustline_height : float
        Height of thrustline above reference
    velocity : float
        Flight velocity (m/s)
    thrust : float
        Thrust force (N)
    """
    
    # Create operating point
    op_point = asb.OperatingPoint(
        velocity=velocity,
        alpha=0,  # Will be trimmed
        beta=0,
        p=0,
        q=0,
        r=0,
    )
    
    # VLM analysis
    vlm = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=op_point,
        align_trailing_vortices_with_wind=True,
    )
    
    # Calculate forces and moments
    aero_forces = vlm.run()
    
    # Thrust moment arm (distance from CG to thrustline)
    # Assuming thrust is aligned with x-axis and CG is at airplane.xyz_ref
    moment_arm_z = thrustline_height  # Vertical distance from CG
    
    # Thrust-induced pitching moment (positive nose-up)
    # If thrustline is above CG, thrust creates nose-down moment
    thrust_moment = -thrust * moment_arm_z
    
    return {
        'CL': aero_forces['CL'],
        'CD': aero_forces['CD'],
        'Cm': aero_forces['Cm'],
        'thrust_moment': thrust_moment,
        'total_moment': aero_forces['Cm'] * 0.5 * 1.225 * velocity**2 * 10.0 + thrust_moment,
        'L/D': aero_forces['CL'] / aero_forces['CD'] if aero_forces['CD'] > 0 else 0,
    }


def sweep_thrustline_positions():
    """
    Sweep through different thrustline positions and analyze effects
    """
    
    # Range of thrustline heights (relative to wing)
    thrustline_heights = np.linspace(-0.5, 1.5, 15)
    
    velocity = 25  # m/s
    thrust = 150   # N
    alpha_range = np.linspace(-5, 10, 8)
    
    results = {
        'thrustline_heights': [],
        'trim_alpha': [],
        'trim_Cm': [],
        'thrust_moment': [],
        'CL_at_trim': [],
        'CD_at_trim': [],
        'LD_at_trim': [],
    }
    
    print("Analyzing thrustline positions...")
    print("=" * 60)
    
    for height in thrustline_heights:
        print(f"\nThrustline height: {height:.2f} m above reference")
        
        # Create aircraft
        airplane, _ = create_aircraft(height)
        
        # Find approximate trim angle
        best_alpha = 0
        min_moment = float('inf')
        
        for alpha in alpha_range:
            op_point = asb.OperatingPoint(
                velocity=velocity,
                alpha=alpha,
                beta=0,
            )
            
            vlm = asb.VortexLatticeMethod(
                airplane=airplane,
                op_point=op_point,
            )
            
            aero = vlm.run()
            
            # Calculate total moment
            moment_arm_z = height
            thrust_moment = -thrust * moment_arm_z
            q_inf = 0.5 * 1.225 * velocity**2
            S_ref = 10.0  # Approximate wing area
            aero_moment = aero['Cm'] * q_inf * S_ref * 1.0  # times reference chord
            total_moment = aero_moment + thrust_moment
            
            if abs(total_moment) < abs(min_moment):
                min_moment = total_moment
                best_alpha = alpha
                best_results = aero
                best_thrust_moment = thrust_moment
        
        # Store results
        results['thrustline_heights'].append(height)
        results['trim_alpha'].append(best_alpha)
        results['trim_Cm'].append(best_results['Cm'])
        results['thrust_moment'].append(best_thrust_moment)
        results['CL_at_trim'].append(best_results['CL'])
        results['CD_at_trim'].append(best_results['CD'])
        results['LD_at_trim'].append(best_results['CL'] / best_results['CD'])
        
        print(f"  Trim alpha: {best_alpha:.2f}°")
        print(f"  Thrust moment: {best_thrust_moment:.2f} Nm")
        print(f"  CL at trim: {best_results['CL']:.3f}")
        print(f"  L/D at trim: {best_results['CL'] / best_results['CD']:.2f}")
    
    return results


def plot_results(results):
    """
    Create visualization of thrustline position effects
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Effect of Thrustline Position on Aircraft Trim and Performance', 
                 fontsize=14, fontweight='bold')
    
    heights = results['thrustline_heights']
    
    # Plot 1: Trim angle of attack
    axes[0, 0].plot(heights, results['trim_alpha'], 'b-o', linewidth=2, markersize=6)
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].set_xlabel('Thrustline Height Above Reference (m)')
    axes[0, 0].set_ylabel('Trim Angle of Attack (deg)')
    axes[0, 0].set_title('Trim AoA vs Thrustline Position')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Thrust-induced pitching moment
    axes[0, 1].plot(heights, results['thrust_moment'], 'r-s', linewidth=2, markersize=6)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Thrustline Height Above Reference (m)')
    axes[0, 1].set_ylabel('Thrust Pitching Moment (Nm)')
    axes[0, 1].set_title('Thrust-Induced Moment (negative = nose down)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Lift coefficient at trim
    axes[1, 0].plot(heights, results['CL_at_trim'], 'g-^', linewidth=2, markersize=6)
    axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Thrustline Height Above Reference (m)')
    axes[1, 0].set_ylabel('Lift Coefficient at Trim')
    axes[1, 0].set_title('Trim Lift Coefficient')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: L/D at trim
    axes[1, 1].plot(heights, results['LD_at_trim'], 'm-d', linewidth=2, markersize=6)
    axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Thrustline Height Above Reference (m)')
    axes[1, 1].set_ylabel('Lift-to-Drag Ratio')
    axes[1, 1].set_title('L/D at Trim')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/thrustline_analysis.png', dpi=300, bbox_inches='tight')
    print("\n" + "=" * 60)
    print("Plot saved as 'thrustline_analysis.png'")
    
    return fig


def main():
    """
    Main analysis function
    """
    
    print("\n" + "=" * 60)
    print("THRUSTLINE POSITION INVESTIGATION")
    print("=" * 60)
    print("\nThis script analyzes how moving the thrustline above or below")
    print("the wing affects aircraft trim and performance characteristics.")
    print("\nKey effects:")
    print("  - Thrustline above CG → nose-down pitching moment")
    print("  - Thrustline below CG → nose-up pitching moment")
    print("  - Affects trim angle, drag, and overall efficiency")
    print("=" * 60)
    
    # Run the analysis
    results = sweep_thrustline_positions()
    
    # Create visualizations
    plot_results(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 60)
    
    idx_baseline = len(results['thrustline_heights']) // 2
    
    print(f"\nBaseline (thrustline at reference):")
    print(f"  Trim AoA: {results['trim_alpha'][idx_baseline]:.2f}°")
    print(f"  L/D: {results['LD_at_trim'][idx_baseline]:.2f}")
    
    idx_high = -1
    print(f"\nHigh thrustline (+{results['thrustline_heights'][idx_high]:.2f}m):")
    print(f"  Trim AoA: {results['trim_alpha'][idx_high]:.2f}°")
    print(f"  Change: {results['trim_alpha'][idx_high] - results['trim_alpha'][idx_baseline]:.2f}°")
    print(f"  Thrust moment: {results['thrust_moment'][idx_high]:.2f} Nm (nose-down)")
    print(f"  L/D: {results['LD_at_trim'][idx_high]:.2f}")
    
    idx_low = 0
    print(f"\nLow thrustline ({results['thrustline_heights'][idx_low]:.2f}m):")
    print(f"  Trim AoA: {results['trim_alpha'][idx_low]:.2f}°")
    print(f"  Change: {results['trim_alpha'][idx_low] - results['trim_alpha'][idx_baseline]:.2f}°")
    print(f"  Thrust moment: {results['thrust_moment'][idx_low]:.2f} Nm (nose-up)")
    print(f"  L/D: {results['LD_at_trim'][idx_low]:.2f}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()