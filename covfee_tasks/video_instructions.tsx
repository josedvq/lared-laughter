import * as React from 'react'
import {
    Row,
    Col,
    Typography,
    Timeline,
    Button
} from 'antd';
const { Title, Paragraph, Text } = Typography

class VideoInstructionsTask extends React.Component {
    render() {
        return <>
            <Row gutter={16} style={{'marginTop': '20px'}}>
                <Col span={22} offset={1}>
                    <Title>Video detection of laughter</Title>
                    <Paragraph>
                        This task consists in watching 20 videos of approximately 10 seconds each, and rating the enjoyment of the marked target person.
                    </Paragraph>

                    <Paragraph>
                        Pay attention to the movements of the person and rate the enjoyment according to your own perception of how much they are enjoying their conversation.
                    </Paragraph>

                    <Paragraph>
                        For each video, the process is the following:
                    </Paragraph>
                    <Timeline>
                        <Timeline.Item>The video will start with a frame showing the target person enclosed in a bounding box. Make sure to find the target person before watching the video.</Timeline.Item>
                        <Timeline.Item>Play the video, paying attention to the target person.</Timeline.Item>
                        <Timeline.Item>Answer the questions about how much the person is enjoying their conversation.</Timeline.Item>
                        <Timeline.Item>Click next to go to the next segment.</Timeline.Item>
                    </Timeline>
                </Col>
            </Row>
            <Row gutter={16}>
                <Col span={22} offset={1}>
                    <Button>I understand. Start annotating!</Button>
                </Col>
            </Row>
        </>
    }
}

export default VideoInstructionsTask