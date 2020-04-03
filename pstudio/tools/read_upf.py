import numpy as np
import xml.etree.ElementTree as ET
from io import StringIO

class UPF:
    def __init__(self, version):
        self.version = version


def read_upf(filename):
    """Open a UPF file, determine version, and read it"""
    with open(filename) as f:
        xml_file_content = f.read()

    # fix broken XML
    xml_file_content = xml_file_content.replace('&', '&amp;')

    # test if v1
    if xml_file_content.startswith('<PP_INFO>'):
        xml_file_content = '<UPF version="1.0">\n' + xml_file_content + '</UPF>\n'

    # parse the XML file
    #try: 
    root = ET.fromstring(xml_file_content)
    #except ET.ParseError:
    #    pass

    # dispatch to the specific routine
    version = root.attrib["version"]
    upfver = int(version.split(".")[0])
    if upfver == 1:
        return _read_upf_v1(root)
    elif upfver == 2:
        return _read_upf_v2(root)
    else:
        raise RuntimeError('wrong UPF version: {0}'.format(version))


def _read_upf_v1(root):
    """Read a UPF v1 pseudopotential"""
    upf = UPF(1)

    # parse info and header
    upf.info = root.find('PP_INFO').text
    for line in root.find('PP_HEADER').text.split('\n'):
        l = line.split()
        if 'Element' in line:       upf.element = l[0]
        if 'NC' in line:            upf.type = 'NC'
        if 'US' in line:            upf.type = 'US'
        if 'Nonlinear' in line:     upf.nlcc = l[0] == 'T'
        if 'Exchange' in line:      upf.qexc = ' '.join(l[0:4])
        if 'Z valence' in line:     upf.val = float(l[0])
        if 'Max angular' in line:   upf.lmax = int(l[0])
        if 'Number of po' in line:  upf.npoints = int(l[0])
        if 'Number of Wave' in line:
            upf.nwfc = int(l[0])
            upf.nproj = int(l[1])

    # parse mesh
    text = root.find('PP_MESH/PP_R').text
    upf.r = np.array( [float(x) for x in text.split()] )
    text = root.find('PP_MESH/PP_RAB').text
    upf.rab = np.array( [float(x) for x in text.split()] )

    # local potential
    text = root.find('PP_LOCAL').text
    upf.vloc = np.array( [float(x) for x in text.split()] ) / 2.0  # to Hartree

    # atomic wavefunctions
    upf.pswfc = []
    chis = root.find('PP_PSWFC')
    if chis is not None:
        data = StringIO(chis.text)
        nlines = upf.npoints//4
        if upf.npoints % 4 != 0: nlines += 1

        while True:
            line = data.readline()
            if line == '\n': continue
            if line == '': break
            label, l, occ, dummy = line.split()
 
            wfc = []
            for i in range(nlines):
               wfc.extend(map(float, data.readline().split()))
            wfc = np.array(wfc)
            upf.pswfc.append( {'label': label, 'occ': float(occ), 'wfc': wfc} )

    # atomic rho
    upf.atrho = None
    atrho = root.find('PP_RHOATOM')
    if atrho is not None:
        upf.atrho = np.array( [float(x) for x in atrho.text.split()] )

    # TODO: NLCC
    pass

    # TODO: PS_NONLOCAL/BETA, PP_DIJ
    pass

    # TODO: GIPAW data
    pass
    
    return upf


def _read_upf_v2(root):
    """Read a UPF v2 pseudopotential"""
    upf = UPF(2)

    # parse header
    h = root.find('PP_HEADER').attrib
    upf.element = h['element']
    upf.type = h['pseudo_type']
    upf.nlcc = h['core_correction'] == 'true'
    upf.qexc = h['functional']
    upf.val = float(h['z_valence'])
    upf.lmax = int(h['l_max'])
    upf.npoints = int(h['mesh_size'])
    upf.nwfc = int(h['number_of_wfc'])
    upf.nproj = int(h['number_of_proj'])
    upf.v2_header = h.copy()

    # parse mesh
    text = root.find('PP_MESH/PP_R').text
    upf.r = np.array( [float(x) for x in text.split()] )
    text = root.find('PP_MESH/PP_RAB').text
    upf.rab = np.array( [float(x) for x in text.split()] )

    # local potential
    text = root.find('PP_LOCAL').text
    upf.vloc = np.array( [float(x) for x in text.split()] ) / 2.0  # to Hartree

    # atomic wavefunctions
    upf.pswfc = []
    i = 0
    while True:
        i += 1
        chi = root.find('PP_PSWFC/PP_CHI.%i' % (i))
        if chi is None: break

        label = chi.attrib["label"]
        occ = float(chi.attrib["occupation"])
        wfc = [float(x) for x in chi.text.split()]
        wfc = np.array(wfc)
        upf.pswfc.append( {'label': label, 'occ': float(occ), 'wfc': wfc} )

    # TODO: NLCC, ATRHO
    pass

    # TODO: PS_NONLOCAL/BETA, PP_DIJ
    pass

    # TODO: GIPAW data
    pass
           
    return upf


if __name__ == '__main__':
    upf1 = read_upf('C.pbe-tm-gipaw-dc.UPF')
    upf2 = read_upf('Nh.pbe-tm-dc.UPF')

