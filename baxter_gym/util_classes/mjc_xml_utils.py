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


def get_item_from_mesh(name, mesh_file, rgba="1 1 1 1"):
    body = '''
            <body name="free_body_{0}">
                <freejoint name="{0}"/>
                <body name="{0}">
                    <geom type="mesh" mesh="{0}" rgba="{1}"/>
                </body>
            </body>
           '''.format(name, rgba)
    return name, xml.fromstring(body), {'assets': [xml.Element('mesh', {'name':name, 'file':mesh_file})]}


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


def generate_xml(base_file, target_file, items, include_files=[], include_meshes=[], timestep=0.002):
    base_xml = xml.parse(base_file)
    root = base_xml.getroot()
    worldbody = root.find('worldbody')
    contacts = root.find('contact')
    assets = root.find('asset')
    equality = root.find('equality')

    compiler_str = '<compiler coordinate="local" angle="radian" meshdir="{0}" texturedir="textures/" strippath="false" />'.format(baxter_gym.__path__[0]+'/robot_info/meshes')
    compiler_xml = xml.fromstring(compiler_str)
    root.append(compiler_xml)

    option_str = '<option timestep="{0}"  gravity="0 0 -9.81" solver="Newton" noslip_iterations="0"/>'.format(timestep)
    option_xml = xml.fromstring(option_str)
    root.append(option_xml)

    for name, f_name, color in include_meshes:
        items.append(get_item_from_mesh(name, f_name, color))

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
        worldbody.append(xml.fromstring('<include file="{0}" />'.format(f_name)))

    base_xml.write(target_file)
