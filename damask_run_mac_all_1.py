#!/egr/research/CMM/opt/anaconda3/bin/python

import numpy as np
import damask
import yaml
import os
import sys
from scipy.stats import qmc
import pandas as pd

#############################################################################################
#############################################################################################
param_bounds = {
    's': (779e6, 780e6),     #820
    'e': (920e6, 921e6),    # 855
    'f': (1747e6, 1748e6),  #1058
    'l': (947e6, 948e6),    #1036
    'h': (2668e6, 2669e6),#948
    'm': (9010e6, 9011e6),  #926
    'n': (5920e6, 5921e6), #924
    'q': (6655e6, 6656e6) #883
}

n_samples = 1
sampler = qmc.LatinHypercube(d=8)  
sample = sampler.random(n=n_samples)

param_samples = qmc.scale(sample, 
                         [param_bounds['s'][0], param_bounds['e'][0], param_bounds['f'][0], 
                          param_bounds['l'][0], param_bounds['h'][0], param_bounds['m'][0], 
                          param_bounds['n'][0], param_bounds['q'][0]],
                         [param_bounds['s'][1], param_bounds['e'][1], param_bounds['f'][1], 
                          param_bounds['l'][1], param_bounds['h'][1], param_bounds['m'][1], 
                          param_bounds['n'][1], param_bounds['q'][1]])

print(f"使用拉丁超立方采样生成 {n_samples} 个参数组合")

###############################################################################
with open('param_samples.txt', 'w') as f:
    f.write("# 拉丁超立方采样参数组合\n")
    f.write(f"# 总样本数: {n_samples}\n")
    f.write("# 参数说明:\n")
    f.write(f"# s: 范围 [{param_bounds['s'][0]:.2e}, {param_bounds['s'][1]:.2e}]\n")
    f.write(f"# e: 范围 [{param_bounds['e'][0]}, {param_bounds['e'][1]}]\n")
    f.write(f"# f: 范围 [{param_bounds['f'][0]:.2e}, {param_bounds['f'][1]:.2e}]\n")
    f.write(f"# l: 范围 [{param_bounds['l'][0]:.2e}, {param_bounds['l'][1]:.2e}]\n")
    f.write(f"# h: 范围 [{param_bounds['h'][0]}, {param_bounds['h'][1]}]\n")
    f.write(f"# m: 范围 [{param_bounds['m'][0]:.2e}, {param_bounds['m'][1]:.2e}]\n")
    f.write(f"# n: 范围 [{param_bounds['n'][0]:.2e}, {param_bounds['n'][1]:.2e}]\n")
    f.write(f"# q: 范围 [{param_bounds['q'][0]:.2e}, {param_bounds['q'][1]:.2e}]\n")
    f.write("#\n")
    f.write("# 样本编号, s, e, f, l, h, m, n, q\n")
    
    # 写入每个参数组合
    for i, params in enumerate(param_samples):
        s, e, f_val, l, h, m, n, q = params
        f.write(f"{i+1}, {s:.6e}, {e:.6f}, {f_val:.6e}, {l:.6e}, {h:.6f}, {m:.6e}, {n:.6e}, {q:.6e}\n")

print("参数样本已保存到 param_samples.txt")

with open('param_samples_readable.txt', 'w') as f:
    f.write("拉丁超立方采样参数组合\n")
    f.write("=" * 80 + "\n")
    f.write(f"总样本数: {n_samples}\n\n")
    
    f.write("参数范围说明:\n")
    f.write(f"  s: {param_bounds['s'][0]:.2e} 到 {param_bounds['s'][1]:.2e}\n")
    f.write(f"  e: {param_bounds['e'][0]} 到 {param_bounds['e'][1]}\n")
    f.write(f"  f: {param_bounds['f'][0]:.2e} 到 {param_bounds['f'][1]:.2e}\n")
    f.write(f"  l: {param_bounds['l'][0]:.2e} 到 {param_bounds['l'][1]:.2e}\n")
    f.write(f"  h: {param_bounds['h'][0]} 到 {param_bounds['h'][1]}\n")
    f.write(f"  m: {param_bounds['m'][0]:.2e} 到 {param_bounds['m'][1]:.2e}\n")
    f.write(f"  n: {param_bounds['n'][0]:.2e} 到 {param_bounds['n'][1]:.2e}\n")
    f.write(f"  q: {param_bounds['q'][0]:.2e} 到 {param_bounds['q'][1]:.2e}\n\n")
    
    f.write("详细参数组合:\n")
    f.write("-" * 80 + "\n")
    
    for i, params in enumerate(param_samples):
        s, e, f_val, l, h, m, n, q = params
        f.write(f"样本 {i+1}:\n")
        f.write(f"  s = {s:.6e}\n")
        f.write(f"  e = {e:.6f}\n")
        f.write(f"  f = {f_val:.6e}\n")
        f.write(f"  l = {l:.6e}\n")
        f.write(f"  h = {h:.6f}\n")
        f.write(f"  m = {m:.6e}\n")
        f.write(f"  n = {n:.6e}\n")
        f.write(f"  q = {q:.6e}\n")
        f.write("-" * 40 + "\n")

print("易读版参数样本已保存到 param_samples_readable.txt")

# 使用Pandas创建表格格式
df = pd.DataFrame(param_samples, columns=['s', 'e', 'f', 'l', 'h', 'm', 'n', 'q'])
df.index.name = '样本编号'
df.index += 1  # 从1开始编号

# 保存为CSV文件
df.to_csv('param_samples.csv', float_format='%.6e')
print("参数样本已保存为CSV格式: param_samples.csv")

# 在控制台也显示结果
print("\n参数样本详情:")
print(df)

################################################################

# 创建材料配置文件
for i, params in enumerate(param_samples):
    s, e, f_val, l, h, m, n, q = params
    # 转换为Python原生类型
    s_py, e_py, f_py, l_py, h_py, m_py, n_py, q_py = map(float, (s, e, f_val, l, h, m, n, q))
    w_py=s_py*f_py
    # 创建参数标识符
    s_int = f"{s:.2e}".replace('.', '_').replace('+', '')
    e_int = f"{e:.2e}".replace('.', '_').replace('+', '')
    f_int = f"{f_val:.2e}".replace('.', '_').replace('+', '')
    l_int = f"{l:.2e}".replace('.', '_').replace('+', '')
    h_int = f"{h:.2f}".replace('.', '_')
    m_int = f"{m:.2e}".replace('.', '_').replace('+', '')
    n_int = f"{n:.2e}".replace('.', '_').replace('+', '')
    q_int = f"{q:.2e}".replace('.', '_').replace('+', '')
    
    param_id = f"d{s_int}_e{e_int}_f{f_int}_l{l_int}_h{h_int}_m{m_int}_n{n_int}_q{q_int}"
    sample_id = i + 1 
    with open(f'material_{sample_id}_aaaaa_{param_id}.yaml','w') as cfg:
        '''
        homogenization: no homogenization is needed for a fine grid like this
        microstructure: grain constituents information, orientation etc
        phase: more information about material type
        '''
        #homogeization
        homo = {'homogenization':{'direct':{'N_constituents':1,
                                    'mechanical':{'type': 'pass'}}}}
        yaml.dump(homo, cfg)

    #microstructure
###################################################
    #microstructure
        
#        microlist = []
#        for i in range(N):
#            microlist.append({
#                        'homogenization':'SX',
#                        'constituents':[{'v':1.0,
#                                        'O':damask.Rotation.from_random().as_quaternion().tolist(),
#                                        'phase':'A'}],
#                        })
#        yaml.dump({'material':microlist}, cfg)

# 读取文件1.yaml中的参数
        with open('25.yaml', 'r', encoding='utf-8') as f:
             data = yaml.safe_load(f)
             microlist = data.get('material', []) 
             yaml.dump({'material': microlist}, cfg)
######################################################
######################################################
        #phase
        phasedict = {
                        '1': {
                             'lattice': 'hP',
                             'c/a': '1.6333',
                             'mechanical': {
                                        'output': ['F', 'P', 'F_e', 'F_p', 'L_p', 'O','xi_sl','gamma_sl'],
                                        'elastic': { 'type': 'Hooke','C_11': 170.7e9, 'C_12': 90.7e9,'C_13': 60.7e9,'C_33': 180.6e9,'C_44': 42.5e9,},
                                        'plastic': {
                                                'type': 'phenopowerlaw',
                                                'flux': 'False',             
                                                'N_sl': [3,3,0,12],
                                                'dot_gamma_0_sl':[0.001,0.001,0.0,0.001], #[n_py,n_py,0,n_py],
                                                'n_sl': [20,20,1,20],
                                                'h_sl-sl': [1, 1, 5.123, 0.574, 1.123, 1.123, 1],
                                                'a_sl': [2.0,2.0,1.0,2.0], 
                                                'xi_0_sl':[3.49e+8,1.5e+8,0,f_py],           #[3.49e+8,1.5e+8,0,1.107e+9], #[e_py],
                                                'xi_inf_sl': [5.68e+8,1.5e+8,0,3.42e+8],#l_py,                                               
                                                'h_0_sl-sl': [2e+8,2e+8,0,2e+8], #[f_py],
                                                'h_sl-sl':  [+1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0, -1.0,
                                                             -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  1.0,
                                                             +1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
                                                             +1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
                                                             +1.0,  1.0,  1.0,  1.0,  1.0,  1.0]  }}},

                        '2': {
                            'lattice': 'cI',
                            'mechanical': {
                                       'output': ['F', 'P', 'F_e', 'F_p', 'L_p', 'O', 'xi_sl', 'gamma_sl'],
                                       'elastic': { 'type': 'Hooke','C_11': 135.0e+9,'C_12': 113.0e+9,'C_44': 55.9e+9},
                                       'plastic': {
                                               'type': 'phenopowerlaw',
                                               'flux': 'false',
                                               'N_sl': [12],
                                               'dot_gamma_0_sl': 0.001,
                                               'xi_0_sl': s_py,#620
                                               'xi_inf_sl': 333.0e+6,
                                               'h_0_sl-sl': 140e+6,
                                               'n_sl': 15,
                                               'a_sl': 2.25,
                                               'h_sl-sl': [1, 1.4, 1, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4]}}},
                         '3': {
                            'lattice': 'tI',
                            'c/a': 0.5458,
                            'mechanical': {
                                 'output': ['F', 'P', 'F_e', 'F_p', 'L_p', 'O', 'xi_sl', 'gamma_sl'],
                                 'elastic': {'type': 'Hooke','C_11': 180e+9,'C_12': 86.12e+9,'C_13': 86.12e+9,'C_33': 180.7e+9,'C_44': 49.7e+9,'C_66': 49e+9},
                                 'plastic': {
                                         'type': 'phenopowerlaw',
                                         'N_sl': [2, 2, 2, 4, 2, 4, 2, 2, 4, 0, 0, 8],
                                         'n_sl': [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 1, 1, 6.0],
                                         'a_sl': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0],
                                         'h_0_sl-sl': [5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 0.0, 0.0, 5e6],#[20e+7, 20e+7, 20e+7, 20e+7, 20e+7, 20e+7, 20e+7, 20e+7, 20e+7, 0.0, 0.0, 20e+7], #s_py
                                         'xi_0_sl': [e_py, f_py, l_py, f_py, 5.6e+7, 5.1e+7, 7.4e+7, 15.0e+7, 6.6e+7, 0.0, 0.0, m_py],# [8.5e+7, 4.3e+7, 10.4e+7, 4.5e+7, 5.6e+7, 5.1e+7, 7.4e+7, 15.0e+7, 6.6e+7, 0.0, 0.0, 12.0e+7],
                                         'xi_inf_sl': [n_py, h_py, n_py, h_py, l_py, l_py, l_py, l_py, h_py, 0.0, 0.0, q_py],#[11.0e+7, 9.0e+8, 11.0e+7, 9.0e+7, 10.0e+7, 10.0e+7, 10.0e+7, 10.0e+7, 9.0e+7, 0.0, 0.0, 13.0e+7],
                                         'h_sl-sl': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                                                    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                                                    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                                                    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                                                    -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                     1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
                                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                         'dot_gamma_0_sl': [2.6e-8, 2.6e-8, 2.6e-8, 2.6e-8, 2.6e-8, 2.6e-8, 2.6e-8, 2.6e-8, 2.6e-8, 1.0, 1.0, 2.6e-8]
                           }
                          }
}
}



                            
                    
                            
                                                
        yaml.dump({'phase':phasedict}, cfg)

################################



        os.system(f'DAMASK_grid --load tensionX.yaml --geom 1.vti --material material_{sample_id}_aaaaa_{param_id}.yaml > 1_{sample_id}_aaaaa_{param_id}.out')
        res = damask.Result(f'1_tensionX_material_{sample_id}_aaaaa_{param_id}.hdf5')
        res.add_stress_Cauchy()
        res.add_strain()
        strain_vM_data = [] 
        res.add_equivalent_Mises('sigma')
        res.add_equivalent_Mises('epsilon_V^0.0(F)')
        t=res.get('sigma_vM')['increment_4']['1']
        tt=t.shape[0]
        d=res.get('sigma_vM')['increment_4']['2']
        dd=d.shape[0]
        thr=res.get('sigma_vM')['increment_4']['3']
        kk=thr.shape[0]
        I = list(range(0, 160, 4))
        sig_values = [np.average(res.get('sigma_vM').get(f'increment_{i}', {}).get('1', np.nan)) for i in I]
        ig_values = [np.average(res.get('epsilon_V^0.0(F)_vM').get(f'increment_{i}', {}).get('1', np.nan)) for i in I]
        bsig_values = [np.average(res.get('sigma_vM').get(f'increment_{i}', {}).get('3', np.nan)) for i in I]
        big_values = [np.average(res.get('epsilon_V^0.0(F)_vM').get(f'increment_{i}', {}).get('3', np.nan)) for i in I]
        asig_values = [np.average(res.get('sigma_vM').get(f'increment_{i}', {}).get('2', np.nan)) for i in I]
        aig_values = [np.average(res.get('epsilon_V^0.0(F)_vM').get(f'increment_{i}', {}).get('2', np.nan)) for i in I]
        v=tt+dd+kk
        g_values = [(tt* np.average(res.get('epsilon_V^0.0(F)_vM').get(f'increment_{inc}', {}).get('1', [])) + dd* np.average(res.get('epsilon_V^0.0(F)_vM').get(f'increment_{inc}', {}).get('2', []))+kk* np.average(res.get('epsilon_V^0.0(F)_vM').get(f'increment_{inc}', {}).get('3', [])))/ v for inc in I]
        w_values = [(tt* np.average(res.get('sigma_vM').get(f'increment_{inc}', {}).get('1', [])) + dd* np.average(res.get('sigma_vM').get(f'increment_{inc}', {}).get('2', []))+kk* np.average(res.get('sigma_vM').get(f'increment_{inc}', {}).get('3', [])))/ v for inc in I]
        epsilon=g_values
        sigma=w_values
        import pandas as pd
        df = pd.DataFrame(epsilon)
        df.to_csv(f'{sample_id}_epsilon.txt', sep=' ', header=False, index=False)
        fb = pd.DataFrame(sigma)
        fb.to_csv(f'{sample_id}__sigma.txt', sep=' ', header=False, index=False)
        sigma_array = np.array(sigma)
        fa = pd.DataFrame(sigma_array / 1000000)  
        fa.to_csv(f'{sample_id}__sigma_MPa.txt', sep=' ', header=False, index=False)            
        #res.export_VTK()
        # 在代码末尾添加
        hdf5_file = f'1_tensionX_material_{sample_id}_aaaaa_{param_id}.hdf5'
        os.remove(hdf5_file)
        print(f"Deleted {hdf5_file}")   
