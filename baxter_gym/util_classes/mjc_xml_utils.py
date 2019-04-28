import baxter_gym
import xml.etree.ElementTree as xml


MUJOCO_MODEL_Z_OFFSET = -0.665 # -0.706
MUJOCO_MODEL_X_OFFSET = -0.07


def get_param_xml(param):
    x, y, z = param.pose[:, 0]
    y, p, r = param.rotation[:, 0]
    if param._type == 'Cloth':
        free_body = xml.Element('body', {'name':'{0}_free_body'.format(param.name)})
        free_body.append(xml.fromstring('<freejoint name="{0}"/>'.format(param.name)))
        height = param.geom.height
        radius = 0.02 # param.geom.radius
        cloth_body = xml.Element('body', {'name': param.name})
        # cloth_geom = xml.SubElement(cloth_body, 'geom', {'name':param.name, 'type':'cylinder', 'size':"{} {}".format(radius, height), 'rgba':"0 0 1 1", 'friction':'1 1 1'})
        cloth_geom = xml.SubElement(cloth_body, 'geom', {'name': param.name, 'type':'sphere', 'size':"{}".format(radius), 'rgba':"0 0 1 1", 'mass': '0.01'})
        cloth_intertial = xml.SubElement(cloth_body, 'inertial', {'pos':'0 0 0', 'quat':'0 0 0 1', 'mass':'0.1', 'diaginertia': '0.01 0.01 0.01'})
        free_body.append(cloth_body)
        return param.name, free_body, {}

    if param._type == 'Can':
        free_body = xml.Element('body', {'name':'{0}_free_body'.format(param.name)})
        free_body.append(xml.fromstring('<freejoint name="{0}"/>'.format(param.name)))
        height = param.geom.height
        radius = param.geom.radius
        if hasattr(param.geom ,'color'):
            color = param.geom.color
        else:
            color = 'blue'

        rgba = "0 0 1 1"
        if color == 'green':
            rgba = "0 1 0 1"
        elif color == 'red':
            rgba = "1 0 0 1"
        elif color == 'black':
            rgba = "0 0 0 1"
        elif color == 'white':
            rgba = "1 1 1 1"

        can_body = xml.Element('body', {'name': param.name})
        can_geom = xml.SubElement(can_body, 'geom', {'name':param.name, 'type':'cylinder', 'size':"{} {}".format(radius, height), 'rgba':rgba, 'friction':'1 1 1'})
        can_intertial = xml.SubElement(can_body, 'inertial', {'pos':'0 0 0', 'quat':'0 0 0 1', 'mass':'0.1', 'diaginertia': '0.01 0.01 0.01'})
        free_body.append(can_body)
        return param.name, free_body, {'contacts': contacts}

    elif param._type == 'Obstacle': 
        length = param.geom.dim[0]
        width = param.geom.dim[1]
        thickness = param.geom.dim[2]
        if hasattr(param.geom ,'color'):
            color = param.geom.color
        else:
            color = 'grey'

        rgba = "0.5 0.5 0.5 1"
        if color == 'red':
            rgba = "1 0 0 1"
        elif color == 'green':
            rgba = "0 1 0 1"
        elif color == 'blue':
            rgba = "0 0 1 0"
        elif color == 'black':
            rgba = "0 0 0 1"
        elif color == 'white':
            rgba = "1 1 1 1"

        table_body = xml.Element('body', {'name': param.name})
        table_geom = xml.SubElement(table_body, 'geom', {'name':param.name, 
                                                         'type':'box', 
                                                         'size':"{} {} {}".format(length, width, thickness)})
        return param.name, table_body, {'contacts': []}

    elif param._type == 'Basket':
        x, y, z = param.pose[:, 0]
        yaw, pitch, roll = param.rotation[:, 0]
        basket_body = xml.Element('body', {'name':param.name, 
                                  'pos':"{} {} {}".format(x, y, z+MUJOCO_MODEL_Z_OFFSET), 
                                  'euler':'{} {} {}'.format(roll, pitch, yaw), 
                                  'mass': "1"})
        # basket_intertial = xml.SubElement(basket_body, 'inertial', {'pos':"0 0 0", 'mass':"0.1", 'diaginertia':"2 1 1"})
        basket_geom = xml.SubElement(basket_body, 'geom', {'name':param.name, 
                                                           'type':'mesh', 
                                                           'mesh': "laundry_basket"})
        return param.name, basket_body, {'contacts': []}


def get_item_from_mesh(name, mesh_name, mesh_file=None, pos=(0, 0, 0), quat=(1, 0, 0, 0), rgba=(1, 1, 1, 1), mass=1., is_fixed=False):
    if is_fixed:
        body == '''
                <body name="{0}" pos="{2} {3} {4}" quat="{5} {6} {7} {8}">
                    <geom type="mesh" mesh="{1}" rgba="{9}" mass="{10}"/>
                </body>
               '''.format(name, mesh_name, pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3], rgba, mass)     
    else:
        body = '''
                <body name="free_body_{0}">
                    <freejoint name="{0}"/>
                    <body name="{0}" pos="{2} {3} {4}" quat="{5} {6} {7} {8}">
                        <geom type="mesh" mesh="{1}" rgba="{9}" mass="{10}"/>
                    </body>
                </body>
               '''.format(name, mesh_name, pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3], rgba, mass)
               
    if mesh_file is not None:
        new_assets = {'assets': [xml.Element('mesh', {'name':name, 'file':mesh_file})]}
    else:
        new_assets = {}

    return name, xml.fromstring(body), new_assets


def get_item(name, item_type, dims, pos=(0, 0, 0), quat=(1, 0, 0, 0), rgba=(1, 1, 1, 1), mass=1., is_fixed=False):
    size = ''
    for d in dims:
        size += "{0} ".format(d)

    color = ''
    for c in rgba:
        color += "{0} ".format(c)

    if is_fixed:
        body = '''
            <body name="{0}" pos="{2} {3} {4}" quat="{5} {6} {7} {8}">
                <geom type="{1}" size="{9}" rgba="{10}" mass="{11}"/>
            </body>
           '''.format(name, item_type, pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3], size, color, mass)
    else:
        body = '''
                <body name="free_body_{0}">
                    <freejoint name="{0}"/>
                    <body name="{0}" pos="{2} {3} {4}" quat="{5} {6} {7} {8}">
                        <geom type="{1}" size="{9}" rgba="{10}" mass="{11}"/>
                    </body>
                </body>
               '''.format(name, item_type, 0, 0, 0, 1, 0, 0, 0, size, color, mass)

    return name, xml.fromstring(body), {}


def get_table():
    body = '''
            <body name="table" pos="0.5 0 -0.4875" euler="0 0 0">
              <geom name="table" type="box" size="0.75 1.5 0.4375" />
            </body>
           '''
    xml_body = xml.fromstring(body)
    contacts = [
        xml.Element('exclude', {'body1': 'table', 'body2': 'pedestal'}),
        xml.Element('exclude', {'body1': 'table', 'body2': 'torso'}),
        xml.Element('exclude', {'body1': 'table', 'body2': 'right_arm_mount'}),
        xml.Element('exclude', {'body1': 'table', 'body2': 'left_arm_mount'}),
        xml.Element('exclude', {'body1': 'table', 'body2': 'base'}),
    ]
    return 'table', xml_body, {'contacts': contacts}


def get_deformable_cloth(width, length, spacing=0.1, radius=0.2, pos=(1.,0.,1.)):
    body =  '''
                <body name="B0_0" pos="{0} {1} {2}">
                    <freejoint />
                    <composite type="cloth" count="{3} {4} 1" spacing="{5}" flatinertia="0.01">
                        <joint kind="main" armature="0.01"/>
                        <skin material="cloth" texcoord="true" inflate="0.005" subgrid="2" />
                        <geom type="sphere" size="{6}" mass="0.005"/>
                    </composite>
                </body>\n
                '''.format(pos[0], pos[1], pos[2], length, width, spacing, radius)

    xml_body = xml.fromstring(body)
    texture = '<texture name="cloth" type="2d" file="cloth_4.png" />'
    xml_texture = xml.fromstring(texture)

    material = '<material name="cloth" texture="cloth" shininess="0.0" />'
    xml_material = xml.fromstring(material)

    return 'B0_0', xml_body, {'assets': [xml_texture, xml_material]}


def get_rope(length, spacing=0.1, radius=0.2, pos=(1.,0.,1.), color="0.8 0.2 0.1 1"):
    body =  '''
                <body name="B0_0" pos="{0} {1} {2}">
                    <freejoint />
                    <composite type="rope" count="{3} {4} 1" spacing="{5}" flatinertia="0.01">
                        <joint kind="main" armature="0.01"/>
                        <geom type="sphere" size="{6}" mass="0.005" rgba="{7}"/>
                    </composite>
                </body>\n
                '''.format(pos[0], pos[1], pos[2], length, width, spacing, radius, color)

    xml_body = xml.fromstring(body)
    texture = '<texture name="cloth" type="2d" file="cloth_4.png" />'
    xml_texture = xml.fromstring(texture)

    material = '<material name="cloth" texture="cloth" shininess="0.0" />'
    xml_material = xml.fromstring(material)

    return 'B0_0', xml_body, {'assets': [xml_texture, xml_material]}


def generate_xml(base_file, target_file, items=[], include_files=[], include_items=[], timestep=0.002):
    '''
    Parameters:
        base_file: path to where the base environment is defined
        target_file: path to write the new environment to
        items: list of (name, item_body, tag_dict) where:
            name: item's name in the environment
            item_body: mjcf definition of the item body
            tag_dict: additional dicitonary of mjcf elements to include; keys are mjcf sections (one of assets, contacts, equality), values are mjcf elements to add
        include_files: list of file names to include. Currently supports mjcf files (parsed into bodies) and stl files (parsed into mesh elements)
        include_items: dictionary with keys (name, mesh_name, pos, quat, color) where:
            name: name of item to be added
            type: type of the item to be added (mesh, sphere, box)
            mesh_name / dims: name of mesh to use as defined in the environment if mesh, otherwise dimensions of the item
            pos (optional): 3D position of the item's origin as (x, y, z)
            quat (optional): 4D quaternion representing the item's orientation ((1, 0, 0, 0) is unrotated)
            color (optional): rgba value for the item
        timestep: float representing the time delta mujoco takes at every step; use default unless the simulation is too slow (warning: large values lead to instability)
    '''
    base_xml = xml.parse(base_file)
    root = base_xml.getroot()
    worldbody = root.find('worldbody')
    contacts = root.find('contact')
    assets = root.find('asset')
    equality = root.find('equality')

    compiler_str = '<compiler coordinate="local" angle="radian" meshdir="{0}" texturedir="textures/" strippath="false" />'.format(baxter_gym.__path__[0]+'/')
    compiler_xml = xml.fromstring(compiler_str)
    root.append(compiler_xml)

    option_str = '<option timestep="{0}"  gravity="0 0 -9.81" integrator="Euler" solver="Newton" noslip_iterations="0"/>'.format(timestep)
    option_xml = xml.fromstring(option_str)
    root.append(option_xml)

    for item_dict in include_items:
        name = item_dict["name"]
        item_type = item_dict["type"]
        is_fixed = item_dict.get("is_fixed", False)
        mass = item_dict.get("mass", 0.05)
        pos = item_dict.get("pos", (0, 0, 0))
        quat = item_dict.get("quat", (1, 0, 0, 0))
        rgba = item_dict.get("rgba", (1, 1, 1, 1))

        if item_type == "mesh":
            mesh_name = item_dict["mesh_name"]   
            items.append(get_item_from_mesh(name, mesh_name=mesh_name, pos=pos, quat=quat, rgba=rgba, mass=mass, is_fixed=is_fixed))
        else:
            dims = item_dict["dimensions"]   
            items.append(get_item(name, item_type=item_type, dims=dims, pos=pos, quat=quat, rgba=rgba, mass=mass, is_fixed=is_fixed))


    for name, item_body, tag_dict in items:
        worldbody.append(item_body)
        if 'contacts' in tag_dict:
            for contact in tag_dict['contacts']:
                contacts.append(contact)
        if 'assets' in tag_dict:
            for asset in tag_dict['assets']:
                assets.append(asset)
        if 'equality' in tag_dict:
            for eq in tag_dict['equality']:
                equality.append(eq)

    for f_name in include_files:
        if f_name.lower().endswith('.mjcf'):
            with open(f_name, 'r+') as f:
                data = f.read()
            elem = xml.fromstring(data)

            compiler = elem.find('compiler')
            size = elem.find('size')
            options = elem.find('options')
            if compiler is not None:
                elem.remove(compiler)

            if size is not None:
                elem.remove(size)

            if options is not None:
                elem.remove(options)

            body = elem.find('worldbody')
            if body is not None:
                body.tag = 'body' # Switch from woldbody to body
            else:
                body = elem.find('body')

            name = elem.get('model')
            body.set('name', name)
            new_assets = elem.find('assets')
            mesh = new_assets.find('mesh')
            mesh_file = mesh.get('file')

            path = f_name.rsplit('/', 1)[0]
            mesh.set('file', path+'/'+mesh_file)

            worldbody.append(body)
            assets.append(list(new_assets))

        elif f_name.lower().endswith('.stl'):
            stripped_path = f_name.split('.')[0]
            mesh_name = stripped_path.rsplit('/', 1)[-1]
            elem = xml.Element('mesh', {'name':mesh_name, 'file':f_name})
            assets.append(elem)

    base_xml.write(target_file)
