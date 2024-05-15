# # Endurance and P_req calculations
# plane = Seaplane(lat, lon, tz, pdc0,gamma,cd0=0.0145,cs=True,tracking=False,cdtot = 0.025,n_tot=.75,S=0.38,weight=4*9.81,voltage=22.2,capacity=28.82)

# # List Parameters for different wing area values
# S =      [0.65340, 0.79]
# Cd0 =    [0.01487, 0.015]
# Cdtot =  [0.02572, 0.03]
# weight = [weight, 149.169]

# # Create a figure and two subplots
# fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))
# ax1.set_title('Endurance vs Forward Flight Speed')
# ax1.set_xlabel('Forward Flight Speed [m/s]')
# ax1.set_ylabel('Endurance [H]')
# ax2.set_title('Required Power vs Forward Flight Speed')
# ax2.set_xlabel("Forward Flight Speed [m/s]")
# ax2.set_ylabel('Required Power [W]')

# # Get endurance and required power
# for i in range(0,len(S)):
#     E = []
#     P_req = []
#     U = range(5,40)    
#     for v in U:
#         plane.S = S[i]
#         plane.cd0 = Cd0[i]
#         plane.cdtot = Cdtot[i]
#         plane.weight = weight[i]
#         E.append(plane.get_endurance(v,rho))
#         P_req.append(plane.get_required_power(U=v,rho=rho))
#     # Plot endurance
#     label_1 = "S = {0}".format(plane.S)
#     ax1.plot(U, E,label=label_1)
#     # Plot Required Power
#     ax2.plot(U, P_req,label=label_1)