import { h, Component } from 'preact';

class NodeInfo extends Component {
  constructor(props){
    super(props);
  }

  render(){
    const { node } = this.props;
    const data = node.data();
    const { name } = data;
    const type = data.BA_Application_Type;
    const q = encodeURIComponent(name);
    const baStrategy = data.BA_House_Position;
    const operationalStatus = data.IT_Service_Instance_Operational_Status;
    const architect = data.BA_System_Architect;
    const owner = data.BA_Application_Product_Owner;
    const description = data.BA_Description;
    // const ba_description = data.BA_Description;
    const tc = data.Owning_Transaction_Cycle;
    const dept3 = data.Department_Level_3;
    const connections = data.Degree;

    return h('div', { class: 'node-info' }, [
      h('div', { class: 'node-info-name' }, name),
      h('div', { class: 'node-info-type' }, tc  + ' (transaction cycle)'),
      h('div', { class: 'node-info-type' }, dept3 + ' (dept L3)'),
      h('div', { class: 'node-info-type' }, owner + ' (owner)'),
      h('div', { class: 'node-info-more' }, architect  + ' (architect)'),
      h('br', {class: 'node-info-more'}),
      h('div', { class: 'node-info-more' }, type + ' (app type)'),
      h('div', { class: 'node-info-more' }, baStrategy + ' (strategy)'),
      h('div', { class: 'node-info-more' }, operationalStatus + ' (status)'),
      h('br', {class: 'node-info-more'}),
      h('div', { class: 'node-info-more', style: {maxWidth: '500px'} }, description + ' (description)'),
      h('div', { class: 'node-info-more', style: {maxWidth: '500px'} }, connections + ' (connections)'),
      h('br', {class: 'node-info-more'}),
      h('div', { class: 'node-info-more' }, [
        h('a', { target: '_blank', href: `https://www.google.com/search?q=${q}` }, 'Search')
      ])
    ]);
  }
}

export default NodeInfo;
export { NodeInfo };
