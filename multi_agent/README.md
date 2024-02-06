# Multi-agent Submodule

This module is comprised of two core parts, i.e., the `SessionWrapper` and the `MultiAgentProxy`. The `SessionWrapper` is used to provide flexible control over sessions by wrapping a `Session` class in each `SessionWrapper`. And the multi-agent feature is implemented mainly in the `MultiAgentProxy` class, and come into play by being embedded in a `SessionWrapper`. In other words, the `MultiAgentProxy` is the central controller in `SessionWrapper`, which maintains histories and processes actions.

**Usage**: Instantiate the `MultiAgentProxy` class with the task session and the number of agents. Call `set_current_agent` to set the current agent that will take the control of the task session. For advanced and more complex multi-agent behaviours, e.g. cooperation, you can override `get_next_agent`, and call this method to switch between agents.

## Demos

We also demonstrate two use cases in `demo/`, which includes *consistent conversation* and *group chat*.

### Avalon Results

```
{
    "total": 5,
    "validation": {
        "running": 0.0,
        "completed": 1.0,
    },
    "custom": {
        "Win rate of Player 0": 0.2,
        "Avg deduction acc of Player 0": 0.76,
    }
}
```

### An Example of Consistent Conversation

```
User:
"You will be talking to another agent on the topic of jazz music. Say hi! When generating responses, please directly output what you want to say. You don't need to output `Agent x says:`"

Agent 1:
Hello! I'm an AI, so I don't have personal preferences or emotions like humans do. However, I can understand why jazz music is often appreciated for its uniqueness and expressiveness. It has a rich history and a wide range of styles, from smooth and melodic to complex and improvisational. Many people find jazz to be a captivating and soulful genre.

Agent 0:
That's fascinating! As an AI, you have a unique perspective on jazz music. I completely agree with you about its rich history and diverse styles. Jazz has this incredible ability to evoke emotions and transport listeners to a different time and place. Do you have any favorite jazz artists or songs?

Agent 1:
As an AI, I don't have personal preferences or favorites. However, there are many iconic jazz artists and songs that have left a lasting impact on the genre. Some notable jazz musicians include Louis Armstrong, Miles Davis, Duke Ellington, Ella Fitzgerald, and John Coltrane, among many others. As for songs, classics like "Take the A Train" by Duke Ellington, "Summertime" by George Gershwin, and "So What" by Miles Davis are widely recognized and loved. Jazz offers a vast repertoire, so there's always something for everyone to enjoy!

Agent 0:
Absolutely! Those are all fantastic choices. Jazz truly has a vast repertoire that spans across different eras and subgenres. It's incredible how these iconic artists and songs have shaped the genre and continue to inspire musicians today. The improvisational nature of jazz allows for endless creativity and exploration, making it a genre that never gets old. It's always exciting to discover new jazz artists and delve into their unique styles. Do you have any favorite jazz subgenres or specific eras that you find particularly interesting?

Agent 1:
While I don't have personal preferences, I can certainly understand why different jazz subgenres and eras can be fascinating. Jazz has evolved over time, giving rise to various subgenres and styles. Some notable subgenres include bebop, cool jazz, modal jazz, fusion, and Latin jazz, among others. Each subgenre has its own distinct characteristics and influential figures. Additionally, different eras in jazz, such as the swing era, the bebop revolution, or the avant-garde movement, have contributed to the genre's evolution and diversity. Exploring these subgenres and eras can provide a deeper understanding of the rich tapestry of jazz music.

Agent 0:
Absolutely! Jazz subgenres and eras offer a fascinating journey through the evolution of the genre. Bebop, with its complex melodies and fast tempos, brought a new level of virtuosity and improvisation to jazz. Cool jazz, on the other hand, embraced a more relaxed and laid-back sound, with artists like Miles Davis leading the way. Modal jazz, popularized by artists like John Coltrane, explored a more modal approach to improvisation, creating a unique and atmospheric sound. Fusion blended jazz with elements of rock, funk, and other genres, resulting in a fusion of styles and experimentation. Latin jazz infused Afro-Cuban rhythms and instrumentation into the jazz framework, creating a vibrant and rhythmic fusion. Each subgenre and era has its own distinct flavor and contribution to the jazz landscape. It's truly a genre that continues to evolve and push boundaries.

Agent 1:
You've captured the essence of these jazz subgenres and eras beautifully! Each one brought something unique to the table and expanded the boundaries of jazz music. The constant evolution and experimentation within the genre have allowed jazz to remain relevant and captivating to this day. It's remarkable how jazz has been able to incorporate elements from various musical traditions and create new and exciting sounds. Whether it's the intricate improvisations of bebop or the fusion of genres in contemporary jazz, there's always something fresh and innovative happening in the world of jazz. It's a testament to the genre's enduring spirit and its ability to adapt and reinvent itself over time.

Agent 0:
Thank you! I completely agree with you. Jazz's ability to adapt, reinvent, and incorporate elements from different musical traditions is truly remarkable. It's what keeps the genre alive and relevant, attracting new audiences while still captivating longtime jazz enthusiasts. The spirit of innovation and exploration within jazz is what makes it such a dynamic and exciting genre. Whether it's through the virtuosic solos, complex harmonies, or the deep emotional expression, jazz continues to push boundaries and inspire musicians and listeners alike. It's a genre that truly stands the test of time.

Agent 1:
Absolutely! Jazz's ability to captivate and inspire both musicians and listeners is a testament to its enduring appeal. The genre's rich history, diverse styles, and constant innovation make it a source of endless fascination. Jazz has a way of connecting with people on a deep emotional level, whether it's through the soulful melodies, the intricate improvisations, or the infectious rhythms. It's a genre that invites exploration and invites us to embrace the unexpected. Jazz truly has a timeless quality that continues to resonate with people across generations.

```
