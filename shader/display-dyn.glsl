in vec2 uv;

uniform vec2 size;
uniform sampler2D trails;
uniform sampler2D terrain;

out vec4 fragColor;

void main() {
  // vec2 px = gl_FragCoord.xy;
  // vec2 s = textureSize(color, 0);
  // vec2 p = px/s;

  vec4 tr = texture(trails, uv);
  vec4 te = texture(terrain, uv);

  fragColor = max(tr, te*te);
}
